"""
RouterAgentLoop: Agent loop with learned routing decisions.

Each prompt runs its own async loop:
  1. Small model generates a chunk (64 tokens) via server
  2. Server returns routing decision (yield/continue) from the router MLP
  3. If yield: async HTTP to big model, append tokens, back to step 1
  4. If continue: back to step 1
  5. If </answer> or max tokens: done

All prompts run concurrently via asyncio.gather(). When some prompts are
waiting for big model HTTP, others keep generating on the small model —
the vLLM server batches their requests via continuous batching.

response_mask: small=1 (gradient), big=0 (no gradient)
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Constants
SMALL_CHUNK = 64
MAX_TOTAL_TOKENS = 8320
BIG_CHUNK_CAP = 128
MAX_ROUNDS = 20

# Big model server config
BIG_MODEL_HOST = os.getenv("SPLITREASON_BIG_MODEL_HOST", "localhost")
BIG_MODEL_PORT = int(os.getenv("SPLITREASON_BIG_MODEL_PORT", "8002"))
BIG_MODEL_NAME = os.getenv(
    "SPLITREASON_BIG_MODEL_NAME",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
)


async def _call_big_model_async(
    prompt: str,
    client: httpx.AsyncClient,
    host: str = BIG_MODEL_HOST,
    port: int = BIG_MODEL_PORT,
    model: str = BIG_MODEL_NAME,
    max_tokens: int = BIG_CHUNK_CAP,
    temperature: float = 0.0,
    max_prompt_chars: int = 60000,
) -> str:
    """Async call to big model vLLM completions API."""
    if len(prompt) > max_prompt_chars:
        prompt = prompt[-max_prompt_chars:]

    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1,
    }
    resp = await client.post(url, json=payload, timeout=300)
    if resp.status_code != 200:
        logger.error(f"Big model HTTP {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def _build_response_mask_from_sources(token_sources: List[str]) -> List[int]:
    """Build mask: 1=small (gradient), 0=big (no gradient)."""
    return [1 if src == "small" else 0 for src in token_sources]


class RouterAgentLoop(AgentLoopBase):
    """
    Agent loop with learned router decisions at chunk boundaries.

    Uses server.generate_and_route() which returns both generated tokens
    and a routing decision (yield to big model or continue with small).
    """

    def __init__(self, config, server_manager, tokenizer):
        super().__init__(config, server_manager, tokenizer)
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length

    async def run(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> AgentLoopOutput:
        """Run the router-based generation loop for a single prompt."""
        # Apply chat template to get prompt token IDs
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
            ),
        )

        request_id = uuid4().hex

        # Per-sequence state
        all_response_ids: List[int] = []
        token_sources: List[str] = []
        routing_positions: List[int] = []
        routing_actions: List[int] = []
        routing_logprobs: List[float] = []
        full_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        done = False
        num_yields = 0

        # Sampling params for small model chunks
        small_params = dict(sampling_params)
        small_params["max_tokens"] = SMALL_CHUNK

        async with httpx.AsyncClient(timeout=None) as http_client:
            for round_num in range(MAX_ROUNDS):
                total_tokens = len(prompt_ids) + len(all_response_ids)
                remain = MAX_TOTAL_TOKENS - total_tokens
                if remain <= 4:
                    break

                # ---- Small model chunk + routing decision ----
                small_params["max_tokens"] = max(1, min(SMALL_CHUNK, remain))

                # Call server — returns GenerateOutput with routing decision
                server = self.server_manager._choose_server(request_id)
                output = await server.generate_and_route.remote(
                    prompt_ids=prompt_ids + all_response_ids,
                    sampling_params=small_params,
                    request_id=request_id,
                )

                new_ids = output.token_ids
                stop_reason = output.stop_reason

                all_response_ids.extend(new_ids)
                token_sources.extend(["small"] * len(new_ids))

                delta = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                full_text += delta

                # Check if small model finished (</answer>, EOS, etc.)
                if stop_reason == "stop" or len(new_ids) == 0:
                    done = True
                    break

                # Record routing decision
                pos = len(all_response_ids) - 1
                routing_positions.append(pos)
                routing_actions.append(output.routing_action)
                routing_logprobs.append(output.routing_logprob)

                if output.routing_action == 1:
                    # ---- Yield to big model ----
                    num_yields += 1

                    try:
                        big_text = await _call_big_model_async(
                            prompt=full_text,
                            client=http_client,
                        )
                    except Exception as e:
                        logger.error(f"Big model call failed: {e}")
                        big_text = ""

                    if big_text:
                        big_ids = self.tokenizer.encode(
                            big_text, add_special_tokens=False,
                        )
                        all_response_ids.extend(big_ids)
                        token_sources.extend(["big"] * len(big_ids))
                        full_text += big_text

                    # Check token limit after big model
                    if len(prompt_ids) + len(all_response_ids) >= MAX_TOTAL_TOKENS - 4:
                        break

                # else: continue (next round generates another small chunk)

        # Truncate to response_length
        all_response_ids = all_response_ids[: self.response_length]
        token_sources = token_sources[: len(all_response_ids)]
        while len(token_sources) < len(all_response_ids):
            token_sources.append("small")

        # Build response mask from token sources
        response_mask = _build_response_mask_from_sources(token_sources)

        # Compute switching count
        num_switches = sum(
            1 for j in range(1, len(routing_actions))
            if routing_actions[j] != routing_actions[j - 1]
        )

        metrics = AgentLoopMetrics()
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=response_mask,
            num_turns=num_yields + 1,
            metrics=metrics,
        )

        # Attach routing data for training (RouterActor reads these)
        output.routing_positions = routing_positions
        output.routing_actions = routing_actions
        output.routing_logprobs = routing_logprobs
        output.num_yields = num_yields
        output.num_switches = num_switches

        return output
