"""
SpeculativeVLLMRollout: Sync SPMD rollout with batch-synchronous speculative reasoning.

Uses batch generate() for both small model and probes, and concurrent HTTP for the
big model. This maximizes GPU utilization on both models by giving each a full batch
of work at once, rather than trickling requests one-at-a-time.

Flow per round:
  1. Batch generate (small model) — all active prompts, stop at <bigmodel> or </answer>
  2. Classify: need_big / continue_small / done
  3. Concurrent big model HTTP — ALL need_big prompts fire simultaneously
  4. Batch probe (small model) — all post-big prompts at once
  5. Classify probe results: close / continue_big / force_close
  6. Repeat until all done or MAX_ROUNDS exceeded

Weight sync: FSDPVLLMShardingManager enters context → syncs FSDP weights →
calls generate_sequences() → exits → sleeps vLLM. All small model generation
and probing uses the weight-synced colocated engine.

response_mask: small=1 (gradient), big=0 (no gradient), tags=1 (small model decision)
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

# Constants from modes/speculative_reasoning_perf.py
SMALL_CHUNK = 64
NUM_PROBE_TOKENS = 6
MAX_TOTAL_TOKENS = 8320
MAX_BIG_SEGMENT = 128
BIG_CHUNK_CAP = 128

# Max rounds of small→big→probe before force-finishing stragglers
MAX_ROUNDS = 20

BIG_OPEN = "<bigmodel>"
BIG_CLOSE = "</bigmodel>"

# Big model server config (env vars)
BIG_MODEL_HOST = os.getenv("SPLITREASON_BIG_MODEL_HOST", "localhost")
BIG_MODEL_PORT = int(os.getenv("SPLITREASON_BIG_MODEL_PORT", "8002"))
BIG_MODEL_NAME = os.getenv(
    "SPLITREASON_BIG_MODEL_NAME",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
)

# Hint text stripped from big-model prompts
BIG_HINT = (
    "You always use <bigmodel>...</bigmodel> to mark parts of the "
    "reasoning process that are important."
)


def _build_response_mask(tokenizer, response_ids: list) -> list:
    """
    Build mask: 1=small model (gradient), 0=big model (no gradient).
    <bigmodel>/</bigmodel> tag tokens are 1 (small model's decision).
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
        if (
            not inside
            and i + L_open <= seqlen
            and response_ids[i : i + L_open] == big_open_ids
        ):
            i += L_open
            inside = True
            continue
        if (
            inside
            and i + L_close <= seqlen
            and response_ids[i : i + L_close] == big_close_ids
        ):
            i += L_close
            inside = False
            continue
        if inside:
            mask[i] = 0
        i += 1

    return mask


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


class SpeculativeVLLMRollout(vLLMRollout):
    """
    vLLMRollout subclass with batch-synchronous speculative reasoning.

    Uses batch generate() for the small model (maximizes GPU utilization)
    and concurrent ThreadPoolExecutor for big model HTTP calls (maximizes
    big model batching via vLLM's continuous batching).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.big_model_host = BIG_MODEL_HOST
        self.big_model_port = BIG_MODEL_PORT
        self.big_model_name = BIG_MODEL_NAME
        self._big_model_pool = ThreadPoolExecutor(max_workers=64)
        logger.info(
            f"SpeculativeVLLMRollout: big model at "
            f"{self.big_model_host}:{self.big_model_port} ({self.big_model_name})"
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        return self._speculative_generate(prompts, **kwargs)

    def _speculative_generate(self, prompts: DataProto, **kwargs) -> DataProto:
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

        # Initialize per-prompt state: accumulated token IDs and text
        all_token_ids = [list(pids) for pids in expanded_prompt_ids]  # includes prompt
        all_texts = [self.tokenizer.decode(pids, skip_special_tokens=False) for pids in expanded_prompt_ids]
        prompt_lengths = [len(pids) for pids in expanded_prompt_ids]
        done = [False] * num_outputs
        big_seg_count = [0] * num_outputs  # tokens in current big segment
        in_big = [False] * num_outputs  # currently inside <bigmodel> segment

        # Sampling params
        if not do_sample:
            base_temperature = 0
            base_top_p = 1.0
            base_top_k = -1
        else:
            base_temperature = self.sampling_params.temperature
            base_top_p = self.sampling_params.top_p
            base_top_k = self.sampling_params.top_k

        for round_num in range(MAX_ROUNDS):
            # ---- Phase 1: Small model batch generate ----
            active_small = [
                i for i in range(num_outputs)
                if not done[i] and not in_big[i]
            ]

            if active_small:
                self._batch_small_generate(
                    active_small, all_token_ids, all_texts, done, in_big,
                    big_seg_count, base_temperature, base_top_p, base_top_k,
                )

            # ---- Phase 2: Big model concurrent HTTP ----
            need_big = [i for i in range(num_outputs) if not done[i] and in_big[i]]

            if need_big:
                self._batch_big_generate(
                    need_big, all_token_ids, all_texts, done, in_big, big_seg_count,
                )

            # ---- Phase 3: Probe batch ----
            need_probe = [i for i in range(num_outputs) if not done[i] and in_big[i]]

            if need_probe:
                self._batch_probe(
                    need_probe, all_token_ids, all_texts, done, in_big,
                    big_seg_count, base_temperature, base_top_p, base_top_k,
                )

            # Check if all done
            if all(done):
                break

            # Check if only stragglers remain (prompts stuck in small model re-generation)
            active = [i for i in range(num_outputs) if not done[i]]
            logger.info(
                f"Round {round_num + 1}: {len(active)} active, "
                f"{sum(in_big[i] for i in active)} in_big, "
                f"{sum(not in_big[i] for i in active)} in_small"
            )

        # Force-finish any remaining
        for i in range(num_outputs):
            if not done[i]:
                if in_big[i]:
                    close_ids = self.tokenizer.encode(BIG_CLOSE, add_special_tokens=False)
                    all_token_ids[i].extend(close_ids)
                    all_texts[i] += BIG_CLOSE
                done[i] = True

        # Build response tensors
        response_list = []
        mask_list = []

        for i in range(num_outputs):
            response_tokens = all_token_ids[i][prompt_lengths[i]:]
            response_tokens = response_tokens[: self.config.response_length]
            response_mask = _build_response_mask(self.tokenizer, response_tokens)
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

        return DataProto.from_dict(batch_dict, non_tensors=non_tensor_batch)

    # ---- Batch phases ----

    def _batch_small_generate(
        self,
        indices: list,
        all_token_ids: list,
        all_texts: list,
        done: list,
        in_big: list,
        big_seg_count: list,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        """Phase 1: Batch generate with small model for all active-small prompts."""
        # Build inputs for batch generate
        prompts_for_gen = []
        idx_map = []  # maps generate output index -> original index

        for i in indices:
            if done[i]:
                continue
            remain = MAX_TOTAL_TOKENS - (len(all_token_ids[i]))
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
            stop=[BIG_OPEN, "</answer>"],
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

            if BIG_OPEN in gen_text:
                in_big[i] = True
                big_seg_count[i] = 0
            elif finish_reason == "stop" or len(gen_tokens) == 0:
                done[i] = True
            elif len(all_token_ids[i]) >= MAX_TOTAL_TOKENS - 4:
                done[i] = True

    def _batch_big_generate(
        self,
        indices: list,
        all_token_ids: list,
        all_texts: list,
        done: list,
        in_big: list,
        big_seg_count: list,
    ):
        """Phase 2: Concurrent big model HTTP for all prompts that hit <bigmodel>."""
        futures = {}
        for i in indices:
            # Build big model prompt: strip tags
            big_prompt = (
                all_texts[i]
                .replace(BIG_HINT, "")
                .replace(BIG_CLOSE, "")
                .replace(BIG_OPEN, "")
            )
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

        # Wait for ALL to complete (they batch on the big model server)
        for future in as_completed(futures):
            i = futures[future]
            try:
                big_text = future.result()
            except Exception as e:
                logger.error(f"Big model call failed for sample {i}: {e}")
                big_text = ""

            if big_text:
                big_ids = self.tokenizer.encode(big_text, add_special_tokens=False)
                all_token_ids[i].extend(big_ids)
                all_texts[i] += big_text
                big_seg_count[i] += len(big_ids)

            # Check if over total token limit
            if len(all_token_ids[i]) >= MAX_TOTAL_TOKENS - 4:
                close_ids = self.tokenizer.encode(BIG_CLOSE, add_special_tokens=False)
                all_token_ids[i].extend(close_ids)
                all_texts[i] += BIG_CLOSE
                in_big[i] = False
                done[i] = True

    def _batch_probe(
        self,
        indices: list,
        all_token_ids: list,
        all_texts: list,
        done: list,
        in_big: list,
        big_seg_count: list,
        temperature: float,
        top_p: float,
        top_k: int,
    ):
        """Phase 3: Batch probe with small model to check for </bigmodel>."""
        prompts_for_probe = []
        idx_map = []

        for i in indices:
            if done[i] or not in_big[i]:
                continue
            prompts_for_probe.append(TokensPrompt(prompt_token_ids=all_token_ids[i]))
            idx_map.append(i)

        if not prompts_for_probe:
            return

        sp = SamplingParams(
            n=1,
            max_tokens=NUM_PROBE_TOKENS,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            detokenize=True,
        )

        outputs = self.inference_engine.generate(
            prompts=prompts_for_probe,
            sampling_params=sp,
            use_tqdm=False,
        )

        for out_idx, output in enumerate(outputs):
            i = idx_map[out_idx]
            probe_text = (output.outputs[0].text or "").lstrip()

            if probe_text.startswith(BIG_CLOSE[:4]):
                # Small model wants to close — extract any pre-close text
                pre = probe_text.split(BIG_CLOSE, 1)[0]
                if pre:
                    pre_ids = self.tokenizer.encode(pre, add_special_tokens=False)
                    all_token_ids[i].extend(pre_ids)
                    all_texts[i] += pre

                # Append </bigmodel>
                close_ids = self.tokenizer.encode(BIG_CLOSE, add_special_tokens=False)
                all_token_ids[i].extend(close_ids)
                all_texts[i] += BIG_CLOSE
                in_big[i] = False
                big_seg_count[i] = 0
            else:
                # Small model doesn't want to close yet
                if big_seg_count[i] >= MAX_BIG_SEGMENT:
                    # Force close — too many big tokens
                    close_ids = self.tokenizer.encode(BIG_CLOSE, add_special_tokens=False)
                    all_token_ids[i].extend(close_ids)
                    all_texts[i] += BIG_CLOSE
                    in_big[i] = False
                    big_seg_count[i] = 0
                # else: stays in_big, will get another big model chunk next round

            # Check total length
            if len(all_token_ids[i]) >= MAX_TOTAL_TOKENS - 4:
                if in_big[i]:
                    close_ids = self.tokenizer.encode(BIG_CLOSE, add_special_tokens=False)
                    all_token_ids[i].extend(close_ids)
                    all_texts[i] += BIG_CLOSE
                    in_big[i] = False
                done[i] = True
