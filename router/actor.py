"""
RouterActor: DataParallelPPOActor subclass that adds REINFORCE loss
for the learned router alongside the standard PPO policy loss.

The base model trains normally via PPO (or is frozen with LR=0).
The router trains via REINFORCE: -advantage * log π_router(action | hidden).
"""

import torch
import torch.nn as nn

from verl.workers.actor.dp_actor import DataParallelPPOActor


class RouterActor(DataParallelPPOActor):
    """
    Extends DataParallelPPOActor with auxiliary REINFORCE loss for the router.

    The router_head is a small MLP (~200K params) that makes binary
    yield/continue decisions. Its loss is computed from:
    - routing_positions: token positions where decisions were made
    - routing_actions: 0=continue, 1=yield
    - routing_logprobs: log π(action | hidden) at decision time
    - advantages: from GRPO (shared with base policy)
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        router_head: nn.Module = None,
        router_optimizer: torch.optim.Optimizer = None,
        router_loss_coef: float = 1.0,
    ):
        super().__init__(config, actor_module, actor_optimizer)
        self.router_head = router_head
        self.router_optimizer = router_optimizer
        self.router_loss_coef = router_loss_coef

    def _compute_router_loss(self, non_tensor_data: dict, advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute REINFORCE loss for the router using rollout-time log-probs.

        Uses the log π(action | hidden) recorded during rollout rather than
        recomputing from hidden states (which would require storing full
        hidden states through the training forward pass).

        This is the standard REINFORCE estimator:
            L = -1/K Σ advantage_i * Σ_k log π(a_k | h_k)

        Args:
            non_tensor_data: dict containing routing_positions, routing_actions,
                           routing_logprobs from rollout
            advantages: (bs,) or (bs, response_len) tensor of advantages

        Returns:
            Scalar router loss tensor
        """
        routing_logprobs_list = non_tensor_data.get("routing_logprobs", None)
        if routing_logprobs_list is None:
            return torch.tensor(0.0, device=advantages.device, requires_grad=False)

        # Collapse advantages to per-sequence scalar
        if advantages.dim() > 1:
            # Mean over response length (masked mean would be better but
            # advantages are already masked by veRL's advantage computation)
            advantages_scalar = advantages.mean(dim=-1)  # (bs,)
        else:
            advantages_scalar = advantages

        total_loss = torch.tensor(0.0, device=advantages.device)
        total_decisions = 0

        n = min(len(routing_logprobs_list), advantages_scalar.shape[0])
        for i in range(n):
            logprobs = routing_logprobs_list[i]
            if not isinstance(logprobs, (list, tuple)) or len(logprobs) == 0:
                continue

            # Sum of log-probs for this sequence
            logprob_sum = sum(logprobs)
            adv = advantages_scalar[i].item() if advantages_scalar.dim() > 0 else advantages_scalar.item()

            total_loss = total_loss + (-adv * logprob_sum)
            total_decisions += len(logprobs)

        if total_decisions > 0:
            total_loss = total_loss / total_decisions

        return total_loss

    def update_policy(self, data):
        """
        Override parent's update_policy to add router REINFORCE loss.

        The router loss is computed once per mini-batch and added to the
        policy loss before backward(). The router optimizer steps alongside
        the actor optimizer.
        """
        from verl import DataProto
        from verl.utils.torch_functional import (
            logprobs_from_logits,
        )
        from verl.trainer.ppo.core_algos import (
            compute_policy_loss,
            kl_penalty,
            agg_loss,
        )
        from verl.workers.actor.dp_actor import (
            append_to_dict,
            rearrange_micro_batches,
        )
        try:
            from verl.workers.actor.dp_actor import get_policy_loss_fn
        except ImportError:
            get_policy_loss_fn = None
        try:
            from verl.utils.torch_functional import get_device_id
        except ImportError:
            def get_device_id():
                return torch.cuda.current_device()

        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Extract routing data from non_tensor_batch before splitting
        routing_non_tensor = {}
        for key in ["routing_positions", "routing_actions", "routing_logprobs",
                     "num_yields", "num_switches"]:
            if key in data.non_tensor_batch:
                routing_non_tensor[key] = data.non_tensor_batch[key]

        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch_data in enumerate(dataloader):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * getattr(
                        self, "ulysses_sequence_parallel_size", 1
                    )
                    if has_multi_modal_inputs:
                        raise NotImplementedError("Dynamic BSZ with multi-modal not supported in RouterActor")
                    micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch_data, max_token_len=max_token_len
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    if has_multi_modal_inputs:
                        num_micro_batches = mini_batch_data.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = mini_batch_data.select(select_keys, ["multi_modal_inputs"]).chunk(num_micro_batches)
                    else:
                        micro_batches = mini_batch_data.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                if self.router_optimizer is not None:
                    self.router_optimizer.zero_grad()

                # Compute router REINFORCE loss once per mini-batch using
                # full-batch routing data and mini-batch advantages.
                # This uses rollout-time log-probs so no model forward needed.
                router_loss_value = None
                if routing_non_tensor and self.router_head is not None:
                    # Use the full-batch advantages tensor, sliced for this mini-batch
                    mb_size = self.config.ppo_mini_batch_size
                    mb_start = batch_idx * mb_size
                    mb_advantages = batch["advantages"][mb_start:mb_start + mb_size].to(get_device_id())
                    mb_routing = {}
                    for key in routing_non_tensor:
                        mb_routing[key] = routing_non_tensor[key][mb_start:mb_start + mb_size]
                    router_loss_value = self._compute_router_loss(
                        mb_routing, mb_advantages)

                for micro_data in micro_batches:
                    micro_batch_metrics = {}

                    if isinstance(micro_data, DataProto):
                        micro_data = {**micro_data.batch.to(get_device_id()), **micro_data.non_tensor_batch}
                    elif isinstance(micro_data, dict):
                        for k, v in micro_data.items():
                            if isinstance(v, torch.Tensor):
                                micro_data[k] = v.to(get_device_id())
                            elif k == "multi_modal_inputs" and v is not None:
                                micro_data[k] = [
                                    {kk: vv.to(get_device_id()) for kk, vv in item_dict.items()}
                                    for item_dict in v
                                ]
                            else:
                                micro_data[k] = v
                    else:
                        micro_data = micro_data.to(get_device_id())

                    response_mask = micro_data["response_mask"]
                    old_log_prob = micro_data["old_log_probs"]
                    advantages = micro_data["advantages"]

                    clip_ratio = self.config.clip_ratio
                    clip_ratio_low = (
                        self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                    )
                    clip_ratio_high = (
                        self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio
                    )
                    clip_ratio_c = self.config.get("clip_ratio_c", 3.0)
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        micro_batch=micro_data, temperature=temperature,
                        calculate_entropy=calculate_entropy,
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

                    if self.config.policy_loss.loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio_low,
                            cliprange_high=clip_ratio_high,
                            clip_ratio_c=clip_ratio_c,
                            loss_agg_mode=loss_agg_mode,
                        )
                    else:
                        if get_policy_loss_fn is not None:
                            policy_loss_fn = get_policy_loss_fn(loss_mode)
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                                old_log_prob, log_prob, advantages, response_mask,
                                loss_agg_mode, self.config,
                            )
                        else:
                            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask,
                                cliprange=clip_ratio,
                                cliprange_low=clip_ratio_low,
                                cliprange_high=clip_ratio_high,
                                clip_ratio_c=clip_ratio_c,
                                loss_agg_mode=loss_agg_mode,
                            )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(
                            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                        )
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = micro_data["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob,
                            kl_penalty=self.config.kl_loss_type,
                        )
                        kl_loss = agg_loss(
                            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                        )
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    # ---- Router REINFORCE loss (pre-computed per mini-batch) ----
                    if router_loss_value is not None:
                        policy_loss = policy_loss + self.router_loss_coef * router_loss_value
                        micro_batch_metrics["actor/router_loss"] = router_loss_value.detach().item()

                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * (len(micro_data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update({
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                    })
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                # Step router optimizer if present
                if self.router_optimizer is not None:
                    self.router_optimizer.step()

                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        if self.router_optimizer is not None:
            self.router_optimizer.zero_grad()
        return metrics
