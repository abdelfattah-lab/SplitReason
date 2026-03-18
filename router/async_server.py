"""
RouterAsyncvLLMServer: AsyncvLLMServer subclass with learned routing.

Extends the standard vLLM async server with:
1. A router head (RoutingHead MLP) that lives on the model runner workers
2. A forward hook on the final norm layer to capture hidden states
3. A generate_and_route() method that returns token_ids + routing decision

The router head and hook run INSIDE the worker process (same GPU as the
model), so no hidden state transfer across processes is needed.

Architecture:
  AgentLoop → server.generate_and_route() → AsyncLLM.generate()
                                           → worker: hook captures hidden
                                           → worker: router MLP → (action, logprob)
                                           ← (token_ids, stop_reason, action, logprob)

Weight sync: update_router_weights() sends new state_dict to workers.
"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ray
import torch
from omegaconf import DictConfig
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput

from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    AsyncvLLMServer,
    _get_model_runner_workers,
)

logger = logging.getLogger(__name__)

# Big model server config (env vars)
BIG_MODEL_HOST = os.getenv("SPLITREASON_BIG_MODEL_HOST", "localhost")
BIG_MODEL_PORT = int(os.getenv("SPLITREASON_BIG_MODEL_PORT", "8002"))
BIG_MODEL_NAME = os.getenv(
    "SPLITREASON_BIG_MODEL_NAME",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
)


@dataclass
class GenerateOutput:
    """Structured output from generate / generate_and_route."""
    token_ids: List[int]
    stop_reason: Optional[str] = None  # "stop", "length", or None
    routing_action: int = 0  # 0=continue, 1=yield
    routing_logprob: float = 0.0


# ---- Functions that execute inside the worker process ----

def _worker_setup_router(worker, hidden_dim: int, router_state_bytes: bytes):
    """
    Initialize router head and forward hook inside the worker process.

    Called via worker.execute_method.remote(). `worker` is the vLLM
    worker instance (has self.model_runner.model).
    """
    import torch
    import torch.nn as nn

    # Build router head on the same device as the model
    device = next(worker.model_runner.model.parameters()).device
    dtype = next(worker.model_runner.model.parameters()).dtype

    # Inline RoutingHead to avoid import issues in the worker process
    class _RoutingHead(nn.Module):
        def __init__(self, hd, mid=128):
            super().__init__()
            self.fc1 = nn.Linear(hd, mid)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(mid, 1)

        def forward(self, h):
            return self.fc2(self.act(self.fc1(h)))

    router = _RoutingHead(hidden_dim).to(device=device, dtype=dtype)
    if router_state_bytes:
        state_dict = pickle.loads(router_state_bytes)
        router.load_state_dict(state_dict)
    router.eval()

    worker._router_head = router
    worker._hidden_buffer = {}

    # Register hook on final norm layer
    model = worker.model_runner.model
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        norm_layer = model.model.norm

        def hook_fn(module, inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            worker._hidden_buffer["last"] = hidden.detach()

        worker._router_hook = norm_layer.register_forward_hook(hook_fn)
        return "ok"
    else:
        return "no_norm_layer"


def _worker_route_decision(worker):
    """
    Run the router on the last captured hidden state.
    Returns (action, logprob) tuple.
    """
    import torch

    hidden = worker._hidden_buffer.get("last")
    if hidden is None or not hasattr(worker, "_router_head"):
        return (0, 0.0)

    with torch.no_grad():
        # Take the last token's hidden state
        if hidden.dim() == 2:
            h = hidden[-1:]  # (1, hidden_dim)
        elif hidden.dim() == 3:
            h = hidden[:, -1:, :]  # (1, 1, hidden_dim)
            h = h.squeeze(0)
        else:
            return (0, 0.0)

        logit = worker._router_head(h).squeeze()
        prob = torch.sigmoid(logit)
        action = torch.bernoulli(prob).long().item()

        if action == 1:
            log_p = torch.log(prob + 1e-8).item()
        else:
            log_p = torch.log(1 - prob + 1e-8).item()

    return (action, log_p)


def _worker_update_router(worker, state_bytes: bytes):
    """Update the router head weights in the worker process."""
    import pickle as _pickle

    state_dict = _pickle.loads(state_bytes)
    worker._router_head.load_state_dict(state_dict)
    worker._router_head.eval()
    return "ok"


# ---- Server ----

@ray.remote(num_cpus=1)
class RouterAsyncvLLMServer(AsyncvLLMServer):
    """
    AsyncvLLMServer with learned routing at chunk boundaries.

    After each generate() call, queries the rank-0 worker for a routing
    decision based on the hidden state captured by the forward hook.
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str):
        super().__init__(config, vllm_dp_size, vllm_dp_rank, wg_prefix)
        self._rank0_worker = None
        self._router_initialized = False

    async def init_engine(self):
        """Initialize engine and set up router on the rank-0 worker."""
        await super().init_engine()

        # Look up worker actors — same mechanism as the executor
        try:
            vllm_config = self.engine.vllm_config
            workers = _get_model_runner_workers(vllm_config, init_ray=False)
            self._rank0_worker = workers[0]
            logger.info(f"RouterAsyncvLLMServer: found {len(workers)} model runner workers")
        except Exception as e:
            logger.warning(f"Could not look up model runner workers: {e}")
            return

        # Initialize router head on rank-0 worker
        hidden_dim = self.config.model.get("hidden_size", 1536)

        # Load router checkpoint if specified
        router_state_bytes = b""
        router_path = self.config.get("router_checkpoint", "")
        if router_path:
            from router.model import RoutingHead, load_router
            tmp = RoutingHead(hidden_dim)
            load_router(tmp, router_path)
            router_state_bytes = pickle.dumps(tmp.state_dict())

        result = ray.get(
            self._rank0_worker.execute_method.remote(
                pickle.dumps(_worker_setup_router),
                hidden_dim,
                router_state_bytes,
            )
        )
        self._router_initialized = (result == "ok")
        logger.info(f"Router setup on worker: {result}")

    async def generate(
        self,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
        request_id: str,
    ) -> GenerateOutput:
        """Generate tokens and return structured output with stop reason."""
        max_tokens = self.max_model_len - len(prompt_ids)
        sp = SamplingParams(max_tokens=max_tokens, **sampling_params)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)

        generator = self.engine.generate(
            prompt=prompt, sampling_params=sp, request_id=request_id,
        )

        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output

        assert final_res is not None
        completion = final_res.outputs[0]

        return GenerateOutput(
            token_ids=list(completion.token_ids),
            stop_reason=completion.finish_reason,
        )

    async def generate_and_route(
        self,
        prompt_ids: List[int],
        sampling_params: Dict[str, Any],
        request_id: str,
    ) -> GenerateOutput:
        """
        Generate a small model chunk, then query the worker for a routing
        decision based on the captured hidden state.

        Returns GenerateOutput with routing_action and routing_logprob set.
        """
        output = await self.generate(prompt_ids, sampling_params, request_id)

        if not self._router_initialized or self._rank0_worker is None:
            return output

        # Query the worker for a routing decision
        # The hook captured hidden states during the generate() call above
        try:
            action, logprob = ray.get(
                self._rank0_worker.execute_method.remote(
                    pickle.dumps(_worker_route_decision),
                )
            )
            output.routing_action = action
            output.routing_logprob = logprob
        except Exception as e:
            logger.warning(f"Router decision failed: {e}")

        return output

    async def update_router_weights(self, state_dict_bytes: bytes):
        """Update router weights on the worker. Called during training sync."""
        if self._rank0_worker is None:
            return
        try:
            ray.get(
                self._rank0_worker.execute_method.remote(
                    pickle.dumps(_worker_update_router),
                    state_dict_bytes,
                )
            )
        except Exception as e:
            logger.warning(f"Router weight update failed: {e}")
