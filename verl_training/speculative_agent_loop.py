"""
Speculative Reasoning AgentLoop for veRL.

Ports the state machine from modes/speculative_reasoning_perf.py as a veRL
AgentLoopBase subclass. The small model generates via veRL's colocated vLLM
(self.server_manager); when <bigmodel> is detected, HTTP calls go to an
external 32B vLLM server.

response_mask convention:
  - small model tokens = 1 (gets gradient)
  - big model tokens = 0 (no gradient)
  - <bigmodel> / </bigmodel> tag tokens = 1 (small model's decision to offload)

GPU layout: GPU 0-1 run 32B vLLM server (standalone), GPU 2-7 run veRL
colocated actor+rollout for the 1.5B model.
"""

import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import httpx

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Constants from modes/speculative_reasoning_perf.py
SMALL_CHUNK = 64
STREAM_BUCKET = 8
NUM_PROBE_TOKENS = 6
MAX_TOTAL_TOKENS = 8320  # 4096 + 128 + 4096
MAX_BIG_SEGMENT = 128
BIG_CHUNK_CAP = 32

BIG_OPEN = "<bigmodel>"
BIG_CLOSE = "</bigmodel>"

# External big model server config (defaults, overridable via env vars)
BIG_MODEL_HOST = os.getenv("SPLITREASON_BIG_MODEL_HOST", "localhost")
BIG_MODEL_PORT = int(os.getenv("SPLITREASON_BIG_MODEL_PORT", "8002"))
BIG_MODEL_NAME = os.getenv(
    "SPLITREASON_BIG_MODEL_NAME",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
)


def _build_response_mask(
    tokenizer,
    response_ids: list[int],
) -> list[int]:
    """
    Build a mask over response_ids:
      1 = small model token (gets gradient)
      0 = big model token (no gradient)

    The <bigmodel> and </bigmodel> tag tokens themselves are marked as 1
    because they are the small model's decision to offload.

    Ported from grpo_trainer.py:798-857 (_build_small_token_mask).
    """
    big_open_ids = tokenizer.encode(BIG_OPEN, add_special_tokens=False)
    big_close_ids = tokenizer.encode(BIG_CLOSE, add_special_tokens=False)

    L_open = len(big_open_ids)
    L_close = len(big_close_ids)
    seqlen = len(response_ids)

    mask = [1] * seqlen

    if L_open == 0 or L_close == 0:
        return mask

    inside = False
    i = 0
    while i < seqlen:
        # Match <bigmodel> open tag
        if (
            not inside
            and i + L_open <= seqlen
            and response_ids[i : i + L_open] == big_open_ids
        ):
            # Tag tokens stay 1 (small model decision); flip state
            i += L_open
            inside = True
            continue

        # Match </bigmodel> close tag
        if (
            inside
            and i + L_close <= seqlen
            and response_ids[i : i + L_close] == big_close_ids
        ):
            # Tag tokens stay 1; end big-model span
            i += L_close
            inside = False
            continue

        # Inside big-model span -> mask out
        if inside:
            mask[i] = 0

        i += 1

    return mask


async def _stream_big_model(
    prompt: str,
    client: httpx.AsyncClient,
    host: str = BIG_MODEL_HOST,
    port: int = BIG_MODEL_PORT,
    model: str = BIG_MODEL_NAME,
    max_tokens: int = BIG_CHUNK_CAP,
    temperature: float = 0.0,
):
    """Async generator that yields one token-string at a time from big model."""
    url = f"http://{host}:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "n": 1,
    }
    async with client.stream("POST", url, json=payload, timeout=None) as r:
        async for line in r.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() in ("[DONE]", "data: [DONE]"):
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            choice = obj["choices"][0]
            yield choice.get("delta", {}).get("content") or choice.get("text", "")


@register("speculative_reasoning")
class SpeculativeReasoningAgentLoop(AgentLoopBase):
    """
    Agent loop implementing the speculative reasoning state machine.

    Small model generates in 64-token chunks. When it emits <bigmodel>, the
    big model takes over via streaming HTTP to an external vLLM server. Every
    STREAM_BUCKET big-model tokens, the small model is probed for </bigmodel>
    to hand back control.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # Apply chat template to get prompt token ids
        prompt_ids = await self.apply_chat_template(messages)

        request_id = uuid4().hex
        metrics = {}

        # State machine state
        all_response_ids: list[int] = []
        stage = "small"
        big_seg = 0
        done = False

        # We need the full text for big-model streaming (vLLM completions API
        # takes text, not token IDs). Build it incrementally.
        full_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)

        # For probing, we strip the bigmodel hint from the big-model prompt
        big_hint = (
            "You always use <bigmodel>...</bigmodel> to mark parts of the "
            "reasoning process that are important."
        )

        with simple_timer("generate_sequences", metrics):
            async with httpx.AsyncClient(timeout=None) as http_client:
                while not done:
                    total_tokens = len(prompt_ids) + len(all_response_ids)

                    if stage == "small":
                        remain = MAX_TOTAL_TOKENS - total_tokens
                        if remain <= 4:
                            done = True
                            break

                        # Generate small-model chunk via veRL's managed vLLM
                        small_params = dict(sampling_params)
                        small_params["max_tokens"] = max(1, min(SMALL_CHUNK, remain))

                        output = await self.server_manager.generate(
                            request_id=request_id,
                            prompt_ids=prompt_ids + all_response_ids,
                            sampling_params=small_params,
                        )
                        new_ids = output.token_ids
                        all_response_ids.extend(new_ids)

                        # Decode the new tokens to check for <bigmodel> tag
                        delta = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                        full_text += delta

                        if BIG_OPEN in delta:
                            stage = "big"
                            big_seg = 0
                            continue

                        # Check if generation finished (stop token or empty)
                        if output.stop_reason == "stop" or len(new_ids) == 0:
                            done = True
                        continue

                    if stage == "big":
                        if big_seg >= MAX_BIG_SEGMENT:
                            # Force close the big-model segment
                            close_ids = self.tokenizer.encode(
                                BIG_CLOSE, add_special_tokens=False
                            )
                            all_response_ids.extend(close_ids)
                            full_text += BIG_CLOSE
                            stage = "small"
                            continue

                        # Stream from big model (external vLLM server)
                        # Strip bigmodel tags from the prompt for the big model
                        big_prompt = (
                            full_text.replace(big_hint, "")
                            .replace(BIG_CLOSE, "")
                            .replace(BIG_OPEN, "")
                        )

                        bucket: list[str] = []
                        tokens_this_stream = 0

                        async for tok_str in _stream_big_model(
                            prompt=big_prompt,
                            client=http_client,
                        ):
                            bucket.append(tok_str)
                            tokens_this_stream += 1
                            big_seg += 1

                            # Encode and append the big-model token
                            tok_ids = self.tokenizer.encode(
                                tok_str, add_special_tokens=False
                            )
                            all_response_ids.extend(tok_ids)
                            full_text += tok_str

                            # Probe small model for </bigmodel> every STREAM_BUCKET
                            if len(bucket) >= STREAM_BUCKET:
                                handoff = await self._probe_for_handoff(
                                    request_id=request_id,
                                    prompt_ids=prompt_ids,
                                    response_ids=all_response_ids,
                                    bucket=bucket,
                                    sampling_params=sampling_params,
                                )
                                if handoff is not None:
                                    # Small model wants to close the tag
                                    # Append any prefix before </bigmodel> + the close tag
                                    prefix_text, close_text = handoff
                                    if prefix_text:
                                        prefix_ids = self.tokenizer.encode(
                                            prefix_text, add_special_tokens=False
                                        )
                                        all_response_ids.extend(prefix_ids)
                                        full_text += prefix_text
                                    close_ids = self.tokenizer.encode(
                                        BIG_CLOSE, add_special_tokens=False
                                    )
                                    all_response_ids.extend(close_ids)
                                    full_text += BIG_CLOSE
                                    stage = "small"
                                    break
                                bucket.clear()

                        # If we broke out to "small", continue the outer loop
                        if stage == "small":
                            continue

                        # Stream ended naturally
                        if tokens_this_stream == BIG_CHUNK_CAP:
                            # Server hit max_tokens; keep going in big stage
                            continue

                        # Otherwise, big model hit EOS
                        done = True

        # Truncate to response_length
        all_response_ids = all_response_ids[: self.response_length]

        # Build response mask (small=1, big=0, tags=1)
        response_mask = _build_response_mask(self.tokenizer, all_response_ids)
        response_mask = response_mask[: self.response_length]

        metrics["num_preempted"] = -1
        metrics["tool_calls"] = 0.0

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=all_response_ids,
            response_mask=response_mask,
            num_turns=2,
            metrics=metrics,
        )
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output

    async def _probe_for_handoff(
        self,
        request_id: str,
        prompt_ids: list[int],
        response_ids: list[int],
        bucket: list[str],
        sampling_params: dict[str, Any],
    ) -> Optional[tuple[str, str]]:
        """
        Probe the small model to see if it wants to emit </bigmodel>.

        For each token position in the bucket, construct a prefix and generate
        NUM_PROBE_TOKENS. If any probe starts with "</big", hand off.

        Returns:
            None if no handoff, or (prefix_text, close_text) if handoff detected.
        """
        # Build probe prefixes for each position in the bucket
        # We need to decode the last `bucket` worth of response to build
        # text-based prefixes, but since we work at token level, we probe by
        # generating from the current context.
        #
        # The approach: generate NUM_PROBE_TOKENS from the current full context
        # and check if the output starts with </bigmodel>
        probe_params = dict(sampling_params)
        probe_params["max_tokens"] = NUM_PROBE_TOKENS
        probe_params["temperature"] = 0.7

        probe_output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids + response_ids,
            sampling_params=probe_params,
        )

        probe_text = self.tokenizer.decode(
            probe_output.token_ids, skip_special_tokens=False
        ).lstrip()

        if probe_text.startswith(BIG_CLOSE[:4]):
            # Extract any text before </bigmodel>
            pre = probe_text.split(BIG_CLOSE, 1)[0]
            return (pre, BIG_CLOSE)

        return None
