"""
RouterVLLMRollout: Batch-synchronous rollout with learned router decisions.

Replaces tag-based speculative reasoning with a 2-layer MLP router that
makes binary yield/continue decisions using the small model's hidden states.
Uses batch generate() calls throughout (same proven pattern as
SpeculativeVLLMRollout) — no add_request()/step() complexity.

Flow per round:
  1. Batch generate (small model) — all active prompts, stop at </answer> or EOS
  2. Router decision — extract hidden states via hook, compute p(yield), sample
  3. Concurrent big model HTTP — all sequences where action=yield
  4. Repeat until all done or MAX_ROUNDS exceeded

Compared to SpeculativeVLLMRollout:
  - No probe phase (router MLP replaces 6-token probe generation)
  - No <bigmodel>/</ bigmodel> tags in the sequence
  - response_mask driven by router decisions, not tag parsing

Weight sync: FSDPVLLMShardingManager enters context → syncs FSDP weights →
calls generate_sequences() → exits → sleeps vLLM.

response_mask: small=1 (gradient), big=0 (no gradient)
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import requests
import torch
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import (
    _pre_process_inputs,
    _repeat_interleave,
    vLLMRollout,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Constants (synced with speculative_rollout.py)
SMALL_CHUNK = 64
MAX_TOTAL_TOKENS = 8320
BIG_CHUNK_CAP = 128
MAX_ROUNDS = 20

# Big model server config (env vars)
BIG_MODEL_HOST = os.getenv("SPLITREASON_BIG_MODEL_HOST", "localhost")
BIG_MODEL_PORT = int(os.getenv("SPLITREASON_BIG_MODEL_PORT", "8002"))
BIG_MODEL_NAME = os.getenv(
    "SPLITREASON_BIG_MODEL_NAME",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
)


def _call_big_model(
    prompt_text: str,
    host: str = BIG_MODEL_HOST,
    port: int = BIG_MODEL_PORT,
    model: str = BIG_MODEL_NAME,
    max_tokens: int = BIG_CHUNK_CAP,
    temperature: float = 0.0,
    max_prompt_chars: int = 60000,
) -> str:
    """Synchronous call to big model vLLM completions API."""
    if len(prompt_text) > max_prompt_chars:
        prompt_text = prompt_text[-max_prompt_chars:]

    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt_text,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1,
    }
    resp = requests.post(url, json=payload, timeout=300)
    if resp.status_code != 200:
        logger.error(
            f"Big model HTTP {resp.status_code}: {resp.text[:500]}, "
            f"prompt length: {len(prompt_text)} chars"
        )
        resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def _build_response_mask_from_routing(token_sources: list) -> list:
    """Build mask from token_sources: 1=small (gradient), 0=big (no gradient)."""
    return [1 if src == "small" else 0 for src in token_sources]


class RouterVLLMRollout(vLLMRollout):
    """
    vLLMRollout subclass with learned router decisions at chunk boundaries.

    Uses batch generate() for the small model, a forward hook to capture
    hidden states, and concurrent ThreadPoolExecutor for big model HTTP calls.
    """

    def __init__(self, *args, router_head=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.router_head = router_head
        self.big_model_host = BIG_MODEL_HOST
        self.big_model_port = BIG_MODEL_PORT
        self.big_model_name = BIG_MODEL_NAME
        self._big_model_pool = ThreadPoolExecutor(max_workers=64)
        self._hidden_buffer = {}
        self._hook_handle = None
        logger.info(
            f"RouterVLLMRollout: big model at "
            f"{self.big_model_host}:{self.big_model_port} ({self.big_model_name})"
        )

    def _register_hook(self):
        """Register forward hook on the final norm layer to capture hidden states."""
        if self._hook_handle is not None:
            return

        try:
            model = (
                self.inference_engine.llm_engine
                .model_executor.driver_worker.worker
                .model_runner.model
            )
            if hasattr(model, "model") and hasattr(model.model, "norm"):
                norm_layer = model.model.norm
            else:
                logger.warning("Could not find norm layer, hook not registered")
                return

            def hook_fn(module, input, output):
                # vLLM's RMSNorm may return (hidden_states, residual) tuple
                hidden = output[0] if isinstance(output, tuple) else output
                self._hidden_buffer["last_output"] = hidden.detach()

            self._hook_handle = norm_layer.register_forward_hook(hook_fn)
            logger.info("Registered forward hook on final norm layer")
        except Exception as e:
            logger.warning(f"Failed to register hook: {e}")

    def _remove_hook(self):
        """Remove the forward hook."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        self._register_hook()
        try:
            return self._router_generate(prompts, **kwargs)
        except Exception:
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._remove_hook()

    def _router_generate(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        raw_prompt_ids_list = list(non_tensor_batch.pop("raw_prompt_ids"))

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        # Determine n
        if not do_sample or is_validate:
            n = 1
        else:
            n = self.config.n if hasattr(self.config, "n") else self.sampling_params.n
            if n is None or n < 1:
                n = 1

        # Expand prompts by n
        expanded_prompt_ids = []
        for raw_ids in raw_prompt_ids_list:
            if isinstance(raw_ids, np.ndarray):
                raw_ids = raw_ids.tolist()
            for _ in range(n):
                expanded_prompt_ids.append(list(raw_ids))

        num_outputs = len(expanded_prompt_ids)

        # Initialize per-prompt state
        all_token_ids = [list(pids) for pids in expanded_prompt_ids]
        all_texts = [self.tokenizer.decode(pids, skip_special_tokens=False) for pids in expanded_prompt_ids]
        prompt_lengths = [len(pids) for pids in expanded_prompt_ids]
        done = [False] * num_outputs

        # Token source tracking for response_mask
        token_sources = [[] for _ in range(num_outputs)]  # "small" or "big" per token

        # Router decision tracking
        routing_positions = [[] for _ in range(num_outputs)]
        routing_actions = [[] for _ in range(num_outputs)]    # 0=continue, 1=yield
        routing_logprobs = [[] for _ in range(num_outputs)]   # log π(action | hidden)

        # Sampling params
        if not do_sample:
            base_temperature = 0
            base_top_p = 1.0
            base_top_k = -1
        else:
            base_temperature = self.sampling_params.temperature
            base_top_p = self.sampling_params.top_p
            base_top_k = self.sampling_params.top_k

        # Timing
        t_rollout_start = time.time()
        t_small_total = 0.0
        t_big_total = 0.0
        t_router_total = 0.0
        seq_big_calls = [0] * num_outputs
        final_round = 0

        for round_num in range(MAX_ROUNDS):
            final_round = round_num + 1

            # ---- Phase 1: Small model batch generate ----
            active = [i for i in range(num_outputs) if not done[i]]
            if not active:
                break

            t0 = time.time()
            self._batch_small_generate(
                active, all_token_ids, all_texts, done, token_sources,
                base_temperature, base_top_p, base_top_k,
            )
            t1 = time.time()
            t_small_total += t1 - t0

            # ---- Phase 2: Router decisions ----
            still_active = [i for i in range(num_outputs) if not done[i]]
            if not still_active:
                break

            t2 = time.time()
            yield_indices = self._batch_router_decision(
                still_active, all_token_ids, prompt_lengths,
                routing_positions, routing_actions, routing_logprobs,
            )
            t3 = time.time()
            t_router_total += t3 - t2

            # ---- Phase 3: Big model concurrent HTTP for yield actions ----
            if yield_indices:
                t4 = time.time()
                self._batch_big_generate(
                    yield_indices, all_token_ids, all_texts, done, token_sources,
                )
                t5 = time.time()
                t_big_total += t5 - t4
                for i in yield_indices:
                    seq_big_calls[i] += 1

            n_active = sum(not d for d in done)
            logger.warning(
                f"Round {round_num + 1}: "
                f"small={len(active)} router={len(still_active)} "
                f"yield={len(yield_indices)} active={n_active}/{num_outputs}"
            )

            if all(done):
                break

        # Force-finish remaining
        for i in range(num_outputs):
            if not done[i]:
                done[i] = True

        t_rollout_end = time.time()
        t_rollout_wall = t_rollout_end - t_rollout_start
        logger.warning(
            f"Rollout timing: wall={t_rollout_wall:.1f}s "
            f"small={t_small_total:.1f}s big={t_big_total:.1f}s "
            f"router={t_router_total:.1f}s rounds={final_round} batch={num_outputs}"
        )

        # Build response tensors
        response_list = []
        mask_list = []

        for i in range(num_outputs):
            response_tokens = all_token_ids[i][prompt_lengths[i]:]
            response_tokens = response_tokens[: self.config.response_length]
            sources = token_sources[i][: len(response_tokens)]
            while len(sources) < len(response_tokens):
                sources.append("small")
            response_mask = _build_response_mask_from_routing(sources)
            response_list.append(response_tokens)
            mask_list.append(response_mask)

        device = idx.device
        response = pad_2d_list_to_length(
            response_list, self.pad_token_id, max_length=self.config.response_length
        ).to(device)

        response_mask_tensor = pad_2d_list_to_length(
            mask_list, 0, max_length=self.config.response_length
        ).to(device).to(torch.float32)

        # Expand prompt tensors for n>1
        if n > 1 and do_sample:
            idx = _repeat_interleave(idx, n)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)
            batch_size = batch_size * n
            for key in list(non_tensor_batch.keys()):
                val = non_tensor_batch[key]
                if isinstance(val, np.ndarray):
                    non_tensor_batch[key] = np.repeat(val, n, axis=0)
                elif isinstance(val, list):
                    non_tensor_batch[key] = [v for v in val for _ in range(n)]

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch_dict = {
            "prompts": idx,
            "responses": response,
            "input_ids": seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask_tensor,
        }

        # Compute per-sequence routing stats
        num_yields_arr = np.array(
            [sum(routing_actions[i]) for i in range(num_outputs)], dtype=np.int32
        )
        num_switches_arr = np.zeros(num_outputs, dtype=np.int32)
        for i in range(num_outputs):
            actions = routing_actions[i]
            switches = sum(
                1 for j in range(1, len(actions)) if actions[j] != actions[j - 1]
            )
            num_switches_arr[i] = switches

        # Store routing data in non_tensor_batch
        non_tensor_batch["routing_positions"] = np.array(routing_positions, dtype=object)
        non_tensor_batch["routing_actions"] = np.array(routing_actions, dtype=object)
        non_tensor_batch["routing_logprobs"] = np.array(routing_logprobs, dtype=object)
        non_tensor_batch["num_yields"] = num_yields_arr
        non_tensor_batch["num_switches"] = num_switches_arr
        non_tensor_batch["timing_big_calls"] = np.array(seq_big_calls, dtype=np.int32)
        non_tensor_batch["timing_small_s"] = np.array(
            [t_small_total] * num_outputs, dtype=np.float32
        )
        non_tensor_batch["timing_big_s"] = np.array(
            [t_big_total] * num_outputs, dtype=np.float32
        )

        return DataProto.from_dict(batch_dict, non_tensors=non_tensor_batch)

    # ---- Batch phases ----

    def _batch_small_generate(
        self,
        indices: list,
        all_token_ids: list,
        all_texts: list,
        done: list,
        token_sources: list,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        """Phase 1: Batch generate with small model for all active prompts."""
        prompts_for_gen = []
        idx_map = []

        for i in indices:
            if done[i]:
                continue
            remain = MAX_TOTAL_TOKENS - len(all_token_ids[i])
            if remain <= 0:
                done[i] = True
                continue
            prompts_for_gen.append(TokensPrompt(prompt_token_ids=all_token_ids[i]))
            idx_map.append(i)

        if not prompts_for_gen:
            return

        sp = SamplingParams(
            n=1,
            max_tokens=SMALL_CHUNK,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=["</answer>"],
            include_stop_str_in_output=True,
            detokenize=True,
        )

        outputs = self.inference_engine.generate(
            prompts=prompts_for_gen,
            sampling_params=sp,
            use_tqdm=False,
        )

        for out_idx, output in enumerate(outputs):
            i = idx_map[out_idx]
            gen_tokens = list(output.outputs[0].token_ids)
            gen_text = output.outputs[0].text or ""
            finish_reason = output.outputs[0].finish_reason

            all_token_ids[i].extend(gen_tokens)
            all_texts[i] += gen_text
            token_sources[i].extend(["small"] * len(gen_tokens))

            if finish_reason == "stop" or len(gen_tokens) == 0:
                done[i] = True
            elif len(all_token_ids[i]) >= MAX_TOTAL_TOKENS - 4:
                done[i] = True

    def _batch_router_decision(
        self,
        indices: list,
        all_token_ids: list,
        prompt_lengths: list,
        routing_positions: list,
        routing_actions: list,
        routing_logprobs: list,
    ) -> list:
        """
        Phase 2: Make router decisions for all active sequences.

        Extracts hidden states from the hook buffer (captured during
        the preceding generate() call), runs the router MLP, samples
        Bernoulli actions, and records positions/actions/logprobs.

        Returns list of indices where action=yield (need big model).
        """
        if self.router_head is None:
            return []

        yield_indices = []

        valid_indices = [i for i in indices if len(all_token_ids[i]) > 0]
        if not valid_indices:
            return []

        hidden_states = self._get_hidden_states(valid_indices)

        if hidden_states is None:
            # Fallback: all continue (no yield)
            for i in valid_indices:
                pos = len(all_token_ids[i]) - prompt_lengths[i] - 1
                routing_positions[i].append(max(pos, 0))
                routing_actions[i].append(0)
                routing_logprobs[i].append(0.0)
            return []

        # Run router on hidden states
        router_device = next(self.router_head.parameters()).device
        router_dtype = next(self.router_head.parameters()).dtype
        h = hidden_states.to(device=router_device, dtype=router_dtype)
        logits = self.router_head(h).squeeze(-1)  # (num_active,)
        probs = torch.sigmoid(logits)  # p(yield)

        # Sample actions
        actions = torch.bernoulli(probs).long()

        # Compute log probabilities
        log_probs_yield = torch.log(probs + 1e-8)
        log_probs_continue = torch.log(1 - probs + 1e-8)
        log_pi = actions.float() * log_probs_yield + (1 - actions.float()) * log_probs_continue

        for idx_in_batch, i in enumerate(valid_indices):
            pos = len(all_token_ids[i]) - prompt_lengths[i] - 1
            action = actions[idx_in_batch].item()
            lp = log_pi[idx_in_batch].item()

            routing_positions[i].append(max(pos, 0))
            routing_actions[i].append(action)
            routing_logprobs[i].append(lp)

            if action == 1:
                yield_indices.append(i)

        return yield_indices

    def _get_hidden_states(self, indices: list) -> Optional[torch.Tensor]:
        """
        Extract hidden states for the last token of each sequence.

        Uses the hook buffer from the most recent generate() call.
        During decode, each sequence contributes 1 token per step.
        The hook captures the last step's hidden states, ordered by
        the batch index from the generate() call.
        """
        if "last_output" not in self._hidden_buffer:
            return None

        raw_output = self._hidden_buffer.pop("last_output")

        num_needed = len(indices)

        if raw_output.dim() == 2:
            num_tokens = raw_output.size(0)
            if num_tokens >= num_needed:
                # Take the last num_needed tokens (correspond to our sequences)
                return raw_output[-num_needed:]
            else:
                # Mismatch — pad with zeros
                hidden_dim = raw_output.size(1)
                result = torch.zeros(num_needed, hidden_dim,
                                     device=raw_output.device, dtype=raw_output.dtype)
                result[:num_tokens] = raw_output
                return result
        elif raw_output.dim() == 3:
            # (batch, seq_len, hidden_dim) — take last token per sequence
            return raw_output[:num_needed, -1, :]
        else:
            return None

    def _batch_big_generate(
        self,
        indices: list,
        all_token_ids: list,
        all_texts: list,
        done: list,
        token_sources: list,
    ) -> dict:
        """Phase 3: Concurrent big model HTTP for sequences where router yielded."""
        futures = {}
        submit_times = {}
        for i in indices:
            big_prompt = all_texts[i]
            t_submit = time.time()
            future = self._big_model_pool.submit(
                _call_big_model,
                prompt_text=big_prompt,
                host=self.big_model_host,
                port=self.big_model_port,
                model=self.big_model_name,
                max_tokens=BIG_CHUNK_CAP,
                temperature=0.0,
            )
            futures[future] = i
            submit_times[i] = t_submit

        per_seq_dt = {}

        for future in as_completed(futures):
            i = futures[future]
            t_done = time.time()
            per_seq_dt[i] = t_done - submit_times[i]
            try:
                big_text = future.result()
            except Exception as e:
                logger.error(f"Big model call failed for sample {i}: {e}")
                big_text = ""

            if big_text:
                big_ids = self.tokenizer.encode(big_text, add_special_tokens=False)
                all_token_ids[i].extend(big_ids)
                all_texts[i] += big_text
                token_sources[i].extend(["big"] * len(big_ids))

            # Check total token limit
            if len(all_token_ids[i]) >= MAX_TOTAL_TOKENS - 4:
                done[i] = True

        return per_seq_dt
