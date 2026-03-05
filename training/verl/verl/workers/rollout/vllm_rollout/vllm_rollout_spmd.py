# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import asyncio
import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any, Dict, List, Union
import random
import re
import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest

# vLLM 0.15+ moved SamplingMetadata to vllm.v1.sample.metadata
try:
    from vllm.v1.sample.metadata import SamplingMetadata
except ImportError:
    from vllm.model_executor.sampling_metadata import SamplingMetadata

# vLLM 0.15+ moved WorkerWrapperBase to vllm.v1.worker.worker_base
try:
    from vllm.v1.worker.worker_base import WorkerWrapperBase
except ImportError:
    from vllm.worker.worker_base import WorkerWrapperBase
import copy

from verl import DataProto
from verl.utils.debug import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)
        self.max_model_len = max_model_len

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
                n=1,
                logprobs=0,  # can be set to 0 and let actor to recompute
                max_tokens=config.response_length,
                include_stop_str_in_output=True,
                stop=["</answer>", "</think>\n\n<abstain>"]
            )

        self.tokenizer = tokenizer
        self.allow_hint = config.allow_hint
        if self.allow_hint:
            kwargs["stop"].append("</request>")
            kwargs["stop"].append("</response>")  # Stop if model tries to generate its own response

        # Smart hint selection
        self.hint_selection = config.get("hint_selection", "sequential")
        self.hint_selector = None
        if self.hint_selection == "smart" and self.allow_hint:
            from pipeline.core.hint_selector import HintSelector
            helper_model = config.get("helper_model", None)
            if helper_model:
                self.hint_selector = HintSelector(
                    strategy="smart",
                    helper_model=helper_model,
                    tensor_parallel_size=config.get("helper_tensor_parallel_size", 1),
                    gpu_memory_utilization=config.get("helper_gpu_memory_utilization", 0.9),
                )
                print(f"Smart hint selection enabled (helper: {helper_model})")

        self.n = config.n

        kwargs["detokenize"] = True

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        """import debugpy
        debugpy.listen(("0.0.0.0", 5678))  # or ("127.0.0.1", 5678)
        print("⏸ Waiting for debugger attach...")
        debugpy.wait_for_client()
        import debugpy; debugpy.breakpoint()"""
        if self.allow_hint:
            return self.agentic_generate_sequences(prompts, **kwargs)
        else:
            return self.regular_generate_sequences(prompts, **kwargs)

    def construct_partial_inputs(self, prompt_ids, partial_completion):
        """Construct vLLM input from original token segments (no re-tokenization).

        Returns a dict with prompt_token_ids so vLLM uses exact original tokens,
        avoiding tokenization boundary mismatches between rollout and training.
        """
        if len(partial_completion) == 0:
            token_ids = list(prompt_ids) if not isinstance(prompt_ids, list) else prompt_ids
        else:
            token_ids = []
            for tokens in partial_completion:
                token_ids.extend(tokens)
        return {"prompt_token_ids": token_ids}

    def batched_hint(self, hint_indices, idx_to_last_given, hint_list_dups, decoded_texts=None):
        """Select hints for a batch of prompts.

        Args:
            hint_indices: Indices of prompts requesting hints
            idx_to_last_given: Per-prompt last given hint index (-1 = none)
            hint_list_dups: Per-prompt hint lists
            decoded_texts: Per-prompt decoded generation text (for smart selection)

        Returns:
            List of (hint_text, new_last_given_index) tuples
        """
        if self.hint_selector is not None and decoded_texts is not None:
            # Smart hint selection
            partial_gens = [decoded_texts[idx] for idx in hint_indices]
            hints_batch = [hint_list_dups[idx] for idx in hint_indices]
            last_given_batch = [int(idx_to_last_given[idx]) for idx in hint_indices]

            results = self.hint_selector.select_hints_batch_sync(
                partial_gens, hints_batch, last_given_batch,
            )
            return results

        # Sequential fallback
        hint_results = []
        for curr_idx in hint_indices:
            last_given = int(idx_to_last_given[curr_idx])
            next_idx = last_given + 1
            hints = hint_list_dups[curr_idx]
            if next_idx < len(hints):
                hint_results.append((hints[next_idx], next_idx))
            else:
                hint_results.append(("No more hints available.", last_given))

        return hint_results

    def agentic_loop(self, prompts, sampling_params, hint_list, **kwargs):
        """
        Multi-turn agentic loop for hint-based generation.

        All rollouts can ask for hints (up to max_hints). The model generates,
        can request hints via </request>, receives responses, and continues.
        """
        raw_prompt_ids = _repeat_interleave(prompts.non_tensor_batch["raw_prompt_ids"], sampling_params.n)
        # These are countdown-specific; other tasks may not have them
        targets = _repeat_interleave(prompts.non_tensor_batch.get("target", [None] * len(raw_prompt_ids)), sampling_params.n)
        numbers = [nos for nos in prompts.non_tensor_batch.get("numbers", [None] * len(raw_prompt_ids)) for _ in range(sampling_params.n)]

        num_outputs = len(raw_prompt_ids)
        self.max_hints = self.config.get("max_hints", 6)
        self.max_turns = self.config.get("max_turns", self.max_hints + 1)

        idx_to_partials = [[] for _ in range(num_outputs)]
        idx_to_masks = [[] for _ in range(num_outputs)]
        idx_to_num_hints = [0.0 for _ in range(num_outputs)]
        idx_to_last_given = [-1 for _ in range(num_outputs)]  # Track last given hint index
        idx_to_truncation = [0.0 for _ in range(num_outputs)]
        idx_to_malformed = [False for _ in range(num_outputs)]  # Track malformed requests
        idx_to_response_len = [0 for _ in range(num_outputs)]  # Track cumulative response tokens
        completed_indices = [False for _ in range(num_outputs)]
        hint_list_dups = [h for h in hint_list for _ in range(sampling_params.n)]
        max_response_len = self.config.response_length  # Total response budget across all turns

        # Pattern for valid hint request: exactly <request></request> with nothing inside
        valid_request_pattern = r'<request></request>'
        # Pattern to detect any request tag (for malformed detection)
        any_request_pattern = r'</request>'
        answer_pattern = r'<answer>(.*?)</answer>'

        # Single sampling params for all rollouts (n=1 since expansion already done)
        loop_sampling_params = copy.deepcopy(sampling_params)
        loop_sampling_params.n = 1  # Expansion already done via repeat_interleave

        for attempt in range(self.max_turns):
            tp_rank = vllm_ps.get_tensor_model_parallel_rank()

            # Get all incomplete indices
            if tp_rank == 0:
                curr_indices = [idx for idx, completed in enumerate(completed_indices) if not completed]
                inputs = [self.construct_partial_inputs(raw_prompt_ids[idx], idx_to_partials[idx]) for idx in curr_indices]

                # Early length checks:
                # 1. Prompt length check: prevent vLLM error when prompt > max_model_len
                # 2. Response budget check: skip samples that have used up their response budget
                valid_indices = []
                valid_inputs = []
                min_generation_buffer = 64  # Minimum tokens needed to make generation worthwhile
                for idx, inp in zip(curr_indices, inputs):
                    input_len = len(inp["prompt_token_ids"])
                    remaining_response_budget = max_response_len - idx_to_response_len[idx]

                    # Check 1: prompt too long for model context
                    if input_len > self.max_model_len - min_generation_buffer:
                        completed_indices[idx] = True
                        idx_to_truncation[idx] = 1.0
                        if random.randint(1, 32) == 1:
                            print(f"Early truncation: prompt length {input_len} exceeds max_model_len {self.max_model_len}")
                    # Check 2: response budget exhausted
                    elif remaining_response_budget < min_generation_buffer:
                        completed_indices[idx] = True
                        idx_to_truncation[idx] = 1.0
                        if random.randint(1, 32) == 1:
                            print(f"Early truncation: response budget exhausted ({idx_to_response_len[idx]}/{max_response_len})")
                    else:
                        valid_indices.append(idx)
                        valid_inputs.append(inp)

                curr_indices = valid_indices
                inputs = valid_inputs
                broadcast_data = {'curr_indices': curr_indices, 'inputs': inputs}
            else:
                broadcast_data = None

            broadcast_data = vllm_ps._TP.broadcast_object(broadcast_data, src=0)
            curr_indices, inputs = broadcast_data['curr_indices'], broadcast_data['inputs']

            # Skip if all completed
            if len(curr_indices) == 0:
                break

            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            # Debug: print prompt on turn 2+ to verify hint injection format
            if attempt > 0 and len(inputs) > 0:
                # Always print on turn 2+ (1 in 8 chance to reduce spam)
                if random.randint(1, 8) == 1:
                    print(f"\n{'='*60}")
                    print(f"=== TURN {attempt} PROMPT (1/{len(inputs)} samples) ===")
                    print(f"{'='*60}")
                    decoded_prompt = self.tokenizer.decode(inputs[0]["prompt_token_ids"])
                    print(f"Last 800 chars of prompt:")
                    print(repr(decoded_prompt[-800:]))
                    print(f"{'='*60}\n")

            # Single generate call for all incomplete prompts
            outputs = self.inference_engine.generate(inputs, sampling_params=loop_sampling_params, use_tqdm=True)

            # Process the generated text
            hint_indices = []
            for curr_idx, output in zip(curr_indices, outputs):
                partial_tokens = idx_to_partials[curr_idx]
                partial_masks = idx_to_masks[curr_idx]

                # First append prompt if this is the first generation
                if len(partial_tokens) == 0:
                    partial_tokens.append(output.prompt_token_ids)

                output_text = output.outputs[0].text
                gen_tokens = output.outputs[0].token_ids
                partial_tokens.append(gen_tokens)
                partial_masks.append([1] * len(gen_tokens))

                # Update cumulative response length (model-generated tokens only)
                idx_to_response_len[curr_idx] += len(gen_tokens)

                # Check if cumulative response exceeds budget
                if idx_to_response_len[curr_idx] >= max_response_len:
                    completed_indices[curr_idx] = True
                    idx_to_truncation[curr_idx] = 1.0
                    if random.randint(1, 32) == 1:
                        print(f"Response length {idx_to_response_len[curr_idx]} exceeds max {max_response_len}, marking complete")
                    continue  # Skip further processing for this index

                # Check if model tried to generate its own response (malformed)
                if '</response>' in output_text:
                    idx_to_malformed[curr_idx] = True
                    completed_indices[curr_idx] = True
                    if random.randint(1, 8) == 1:
                        print(f"\n!!! MODEL GENERATED </response> !!!")
                        print(f"Output text (last 500 chars): {repr(output_text[-500:])}")
                    continue

                # Check for any </request> tag in output
                has_request_tag = any_request_pattern in output_text
                valid_request_matches = re.findall(valid_request_pattern, output_text)
                answer_matches = re.findall(answer_pattern, output_text, re.DOTALL)

                # Debug: show what was generated when there's a request tag
                if has_request_tag and random.randint(1, 8) == 1:
                    print(f"\n--- Generated output with request tag (attempt {attempt}) ---")
                    print(f"Valid matches: {len(valid_request_matches)}")
                    print(f"Output ending (last 300 chars): {repr(output_text[-300:])}")

                if has_request_tag:
                    # Check if it's a valid request (exactly <request></request>)
                    if len(valid_request_matches) > 0:
                        # Valid hint request
                        idx_to_num_hints[curr_idx] += 1

                        if idx_to_num_hints[curr_idx] > self.max_hints:
                            # Add the failure message but let model continue to produce an answer
                            hint_result = "\n<response>No more hints available.</response>\n"
                            doc_tokens = list(self.tokenizer(hint_result, add_special_tokens=False)['input_ids'])
                            partial_tokens.append(doc_tokens)
                            partial_masks.append([0] * len(doc_tokens))
                            idx_to_response_len[curr_idx] += len(doc_tokens)

                            # Truncation check - only terminate if we've run out of token budget
                            if self.exceeds_vllm_length(partial_tokens) or idx_to_response_len[curr_idx] >= max_response_len:
                                idx_to_truncation[curr_idx] = 1
                                completed_indices[curr_idx] = True
                        else:
                            hint_indices.append(curr_idx)
                    else:
                        # Malformed request (has </request> but not exactly <request></request>)
                        # Mark as completed - reward function will give 0
                        idx_to_malformed[curr_idx] = True
                        completed_indices[curr_idx] = True
                        if random.randint(1, 64) == 1:
                            print(f"Malformed <request> tag detected in output: {output_text[:200]}...")

                elif len(answer_matches) != 0:
                    completed_indices[curr_idx] = True

                else:
                    # No hint request or answer - model failed to produce valid output
                    # Mark as completed and let reward function score it (will get low score)
                    completed_indices[curr_idx] = True

            # Build decoded texts for smart hint selection
            if tp_rank == 0:
                decoded_texts = {}
                if self.hint_selector is not None:
                    for curr_idx in hint_indices:
                        all_tokens = []
                        for tokens in idx_to_partials[curr_idx][1:]:  # Skip prompt tokens
                            all_tokens.extend(tokens)
                        decoded_texts[curr_idx] = self.tokenizer.decode(all_tokens)

                hint_results = self.batched_hint(hint_indices, idx_to_last_given, hint_list_dups, decoded_texts if decoded_texts else None)
                broadcast_data = {
                    'hint_results': hint_results,
                }
            else:
                broadcast_data = None
            hint_results = vllm_ps._TP.broadcast_object(broadcast_data, src=0)['hint_results'] # broadcast tool call results across tp

            for curr_idx, hint_result in zip(hint_indices, hint_results):
                # hint_result is now (hint_text, new_last_given_index) tuple
                hint_text, new_last_given = hint_result
                idx_to_last_given[curr_idx] = new_last_given

                # Add the hint response in <response>...</response> format, followed by <think> to continue reasoning
                partial_tokens = idx_to_partials[curr_idx]
                partial_masks = idx_to_masks[curr_idx]
                hint_injection_str = "\n<response>" + hint_text + "</response>\n<think>\n"
                doc_tokens = list(self.tokenizer(hint_injection_str, add_special_tokens=False)['input_ids'])

                # Debug: show hint injection
                if random.randint(1, 8) == 1:
                    print(f"\n+++ INJECTING HINT +++")
                    print(f"Hint: {hint_result}")
                    print(f"Injection string: {repr(hint_injection_str)}")
                    print(f"Decoded tokens: {repr(self.tokenizer.decode(doc_tokens))}")
                    print(f"Num tokens: {len(doc_tokens)}")

                partial_tokens.append(doc_tokens)
                partial_masks.append([0] * len(doc_tokens))

                # Track hint response tokens (they count toward total response length)
                idx_to_response_len[curr_idx] += len(doc_tokens)

                # Early truncation: check both vLLM context and response budget
                if self.exceeds_vllm_length(partial_tokens):
                    completed_indices[curr_idx] = True
                    idx_to_truncation[curr_idx] = 1
                elif idx_to_response_len[curr_idx] >= max_response_len:
                    completed_indices[curr_idx] = True
                    idx_to_truncation[curr_idx] = 1

        return idx_to_partials, idx_to_masks, idx_to_num_hints, idx_to_truncation, idx_to_malformed        

    def process_agentic_outputs(self, prompts, idx_to_partials, idx_to_masks, idx_to_truncation, device):
        # Get the vllm response and document mask tokens; pad them
        response = []
        document_mask = []
        decoded_responses = []
        for idx, (partial_tokens, partial_masks) in enumerate(zip(idx_to_partials, idx_to_masks)):
            curr_tokens = []
            curr_masks = []
            for tokens, masks in zip(partial_tokens[1:], partial_masks):
                curr_tokens += tokens
                curr_masks += masks

            num_tokens = len(curr_tokens)
            response.append(curr_tokens[:self.config.response_length])
            document_mask.append(curr_masks[:self.config.response_length])
            decoded_responses.append(self.tokenizer.decode(curr_tokens[:self.config.response_length]))
            if num_tokens > self.config.response_length:
                idx_to_truncation[idx] = 1
            
        response = pad_2d_list_to_length(response, self.pad_token_id).to(device)
        document_mask = pad_2d_list_to_length(document_mask, 0).to(device)
        prompts.non_tensor_batch["completions"] = np.array(decoded_responses, dtype='object')

        return response, document_mask

    def exceeds_vllm_length(self, partial_tokens):
        total_len = sum(len(tokens) for tokens in partial_tokens)
        return total_len >= (self.config.prompt_length + self.config.response_length)


    def agentic_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # Carry over from veRL
        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        hint_list = prompts.non_tensor_batch["hint_exprs"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        """if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        #vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]"""

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        """for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )"""

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            """outputs = self.inference_engine.generate(
                                                    prompts=vllm_inputs,  # because we have already convert it to prompt token id
                                                    sampling_params=self.sampling_params,
                                                    use_tqdm=False,
                                                )"""
            #import debugpy; debugpy.breakpoint()
            idx_to_partials, idx_to_masks, idx_to_num_hints, idx_to_truncation, idx_to_malformed = self.agentic_loop(prompts, self.sampling_params, hint_list=hint_list)

            response, document_mask = self.process_agentic_outputs(prompts, idx_to_partials, idx_to_masks, idx_to_truncation, idx.device)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            

            """response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)
            """
            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(
                        non_tensor_batch["tools_kwargs"], self.sampling_params.n
                    )
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(
                        non_tensor_batch["interaction_kwargs"], self.sampling_params.n
                    )
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt"], self.sampling_params.n
                    )

                if "raw_prompt_ids" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt_ids"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt_ids"], self.sampling_params.n
                    )

                if "hint_exprs" in non_tensor_batch.keys():
                    non_tensor_batch["hint_exprs"] = [h for h in hint_list for _ in range(self.sampling_params.n)]

                if "target" in non_tensor_batch.keys():
                    non_tensor_batch["target"] = [h for h in non_tensor_batch["target"] for _ in range(self.sampling_params.n)]

                if "numbers" in non_tensor_batch.keys():
                    non_tensor_batch["numbers"] = [h for h in non_tensor_batch["numbers"] for _ in range(self.sampling_params.n)]

                if "epsilon" in non_tensor_batch.keys():
                    non_tensor_batch["epsilon"] = _repeat_interleave(
                        non_tensor_batch["epsilon"], self.sampling_params.n
                    )


            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        attention_mask_with_documents = torch.cat([attention_mask, document_mask], dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        """batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": full_attention_mask,
                "document_mask": attention_mask_with_documents,
                "position_ids": position_ids,
                "num_hints": idx_to_num_hints
            },
            batch_size=batch_size,
        )"""

        batch_dict = {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": full_attention_mask,
                "document_mask": attention_mask_with_documents,
                "position_ids": position_ids,
                "num_hints": torch.Tensor(idx_to_num_hints)
            }
        """if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs"""

        return DataProto.from_dict(batch_dict, non_tensors=non_tensor_batch)

    def regular_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")
        
        
        vllm_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=True,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                # NOTE(linjunrong): for multi-turn https://github.com/volcengine/verl/pull/1037
                if "tools_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["tools_kwargs"] = _repeat_interleave(
                        non_tensor_batch["tools_kwargs"], self.sampling_params.n
                    )
                if "interaction_kwargs" in non_tensor_batch.keys():
                    non_tensor_batch["interaction_kwargs"] = _repeat_interleave(
                        non_tensor_batch["interaction_kwargs"], self.sampling_params.n
                    )
                if "raw_prompt" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt"], self.sampling_params.n
                    )
                if "raw_prompt_ids" in non_tensor_batch.keys():
                    non_tensor_batch["raw_prompt_ids"] = _repeat_interleave(
                        non_tensor_batch["raw_prompt_ids"], self.sampling_params.n
                    )

                if "hint_exprs" in non_tensor_batch.keys():
                    non_tensor_batch["hint_exprs"] = [h for h in non_tensor_batch["hint_exprs"] for _ in range(self.sampling_params.n)]
                
                if "target" in non_tensor_batch.keys():
                    non_tensor_batch["target"] = [h for h in non_tensor_batch["target"] for _ in range(self.sampling_params.n)]

                if "numbers" in non_tensor_batch.keys():
                    non_tensor_batch["numbers"] = [h for h in non_tensor_batch["numbers"] for _ in range(self.sampling_params.n)]

                if "epsilon" in non_tensor_batch.keys():
                    non_tensor_batch["epsilon"] = _repeat_interleave(
                        non_tensor_batch["epsilon"], self.sampling_params.n
                    )


            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid

        batch_dict = {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch_dict.update({"rollout_log_probs":rollout_log_probs})

        return DataProto.from_dict(batch_dict, non_tensors=non_tensor_batch)

# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncAgenticRollout(BaseRollout):
    """
    Async-style vLLM rollout for multi-turn hint-based generation.

    Uses the sync LLM engine (compatible with FSDP sharding) but with direct
    LLMEngine.add_request() and step() calls for fine-grained control.

    Each prompt progresses independently - when one finishes and needs a hint,
    it immediately gets added back to the queue without waiting for others.

    Key benefits:
    - No waiting for stragglers - fast prompts complete early
    - Continuous GPU utilization
    - Compatible with FSDP weight syncing (same inference_engine interface)
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """Initialize exactly like vLLMRollout for sharding manager compatibility."""
        super().__init__()

        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.max_hints = config.get("max_hints", 6)
        self.max_turns = config.get("max_turns", self.max_hints + 1)
        self.allow_hint = config.allow_hint
        self.n = config.n

        # Smart hint selection
        self.hint_selection = config.get("hint_selection", "sequential")
        self.hint_selector = None
        if self.hint_selection == "smart" and self.allow_hint:
            from pipeline.core.hint_selector import HintSelector
            helper_model = config.get("helper_model", None)
            if helper_model:
                self.hint_selector = HintSelector(
                    strategy="smart",
                    helper_model=helper_model,
                    tensor_parallel_size=config.get("helper_tensor_parallel_size", 1),
                    gpu_memory_utilization=config.get("helper_gpu_memory_utilization", 0.9),
                )
                print(f"Smart hint selection enabled (helper: {helper_model})")

        tensor_parallel_size = config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            import os
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")

            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs

        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        # Create sync LLM engine - same as vLLMRollout for sharding compatibility
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        # Build default sampling params
        self.sampling_params_kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=config.response_length,
            include_stop_str_in_output=True,
            stop=["</answer>", "</think>\n\n<abstain>", "</request>", "</response>"],  # </response> stops model from generating fake hints
            detokenize=True,
        )

        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                self.sampling_params_kwargs[k] = config.get(k)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences using direct LLMEngine control for async-like behavior."""
        if self.allow_hint:
            return self._agentic_generate(prompts, **kwargs)
        else:
            # Fall back to standard generation for non-hint mode
            return self._simple_generate(prompts, **kwargs)

    def _agentic_generate(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Async-style multi-turn generation using direct LLMEngine control.

        Uses add_request() and step() to process prompts independently.
        When a prompt completes and needs a hint, it's immediately re-queued.
        Uses TokensPrompt to avoid re-tokenization overhead.
        """
        from vllm.inputs import TokensPrompt
        from dataclasses import dataclass

        # Extract data from prompts
        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        hint_list = non_tensor_batch.get("hint_exprs", [[] for _ in range(batch_size)])

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        # Build sampling params
        if not do_sample:
            sampling_params = SamplingParams(
                best_of=1, top_p=1.0, top_k=-1, min_p=0.0, temperature=0, n=1,
                max_tokens=self.config.response_length,
                stop=["</answer>", "</think>\n\n<abstain>", "</request>", "</response>"],
                include_stop_str_in_output=True,
            )
        elif is_validate:
            sampling_params = SamplingParams(
                top_k=self.config.val_kwargs.top_k,
                top_p=self.config.val_kwargs.top_p,
                temperature=self.config.val_kwargs.temperature,
                n=1,
                max_tokens=self.config.response_length,
                stop=["</answer>", "</think>\n\n<abstain>", "</request>", "</response>"],
                include_stop_str_in_output=True,
            )
        else:
            sampling_params = SamplingParams(**self.sampling_params_kwargs)

        # Expand prompts by n_samples
        n_samples = self.n
        raw_prompt_ids = _repeat_interleave(non_tensor_batch["raw_prompt_ids"], n_samples)
        targets = _repeat_interleave(non_tensor_batch["target"], n_samples)
        numbers = [nos for nos in non_tensor_batch["numbers"] for _ in range(n_samples)]
        hint_list_expanded = [h for h in hint_list for _ in range(n_samples)]

        num_outputs = len(raw_prompt_ids)

        # Pre-tokenize common strings to avoid repeated tokenization
        request_end_tokens = list(self.tokenizer('</request>', add_special_tokens=False)['input_ids'])

        # Per-prompt state tracking
        @dataclass
        class PromptState:
            prompt_idx: int
            accumulated_tokens: list  # List of token lists
            accumulated_masks: list   # List of mask lists
            hints: list               # Available hints (as token lists)
            hints_text: list          # Available hints as text strings (for smart selection)
            last_given_index: int     # Last given hint index (-1 = none)
            num_hints_used: int
            target: any
            numbers: any
            completed: bool
            turn: int

        # Initialize states
        states = {}
        for i in range(num_outputs):
            prompt_ids = raw_prompt_ids[i]
            if isinstance(prompt_ids, np.ndarray):
                prompt_ids = prompt_ids.tolist()

            hints = hint_list_expanded[i]
            if isinstance(hints, str):
                import ast
                try:
                    hints = ast.literal_eval(hints)
                except:
                    hints = []

            # Pre-tokenize hints (include <think> to guide model to continue reasoning)
            hint_tokens_list = []
            for hint in hints:
                hint_response = f"\n<response>{hint}</response>\n<think>\n"
                hint_tokens_list.append(
                    list(self.tokenizer(hint_response, add_special_tokens=False)['input_ids'])
                )

            states[i] = PromptState(
                prompt_idx=i,
                accumulated_tokens=[list(prompt_ids)],  # Ensure it's a mutable list
                accumulated_masks=[],
                hints=hint_tokens_list,  # Store pre-tokenized hints
                hints_text=list(hints) if isinstance(hints, list) else [],  # Text hints for smart selection
                last_given_index=-1,
                num_hints_used=0,
                target=targets[i],
                numbers=numbers[i],
                completed=False,
                turn=0,
            )

        # Request ID to prompt index mapping
        request_to_prompt = {}
        next_request_id = 0

        # Get the underlying LLMEngine
        llm_engine = self.inference_engine.llm_engine

        # Add initial requests for all prompts using TokensPrompt
        for i, state in states.items():
            request_id = f"req_{next_request_id}"
            next_request_id += 1
            request_to_prompt[request_id] = i

            llm_engine.add_request(
                request_id=request_id,
                prompt=TokensPrompt(prompt_token_ids=state.accumulated_tokens[0]),
                params=sampling_params,
            )

        hint_pattern = r'</request>'
        answer_pattern = r'<answer>(.*?)</answer>'

        completed_count = 0

        # Progress bar for async generation - only on TP rank 0
        tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        pbar = None
        if tp_rank == 0:
            pbar = tqdm(
                total=num_outputs,
                desc="Async agentic generation",
                unit="prompts",
                dynamic_ncols=True,
            )

        # Main processing loop
        while llm_engine.has_unfinished_requests():
            # Process one step
            step_outputs = llm_engine.step()

            for output in step_outputs:
                if not output.finished:
                    continue

                request_id = output.request_id
                if request_id not in request_to_prompt:
                    continue

                prompt_idx = request_to_prompt.pop(request_id)
                state = states[prompt_idx]

                if state.completed:
                    continue

                # Get generated text and tokens
                generated_text = output.outputs[0].text
                generated_tokens = list(output.outputs[0].token_ids)

                # Add to accumulated
                state.accumulated_tokens.append(generated_tokens)
                state.accumulated_masks.append([1] * len(generated_tokens))

                # Check for patterns - use generated_text for pattern matching
                hint_matches = re.findall(hint_pattern, generated_text, re.DOTALL)
                answer_matches = re.findall(answer_pattern, generated_text, re.DOTALL)

                should_continue = False

                if hint_matches:
                    # Model requested a hint (ends with </request>)
                    state.num_hints_used += 1

                    if state.num_hints_used > self.max_hints:
                        # Exceeded limit - inject message but let model continue to produce an answer
                        limit_tokens = list(self.tokenizer(
                            "\n<response>No more hints available.</response>\n",
                            add_special_tokens=False
                        )['input_ids'])
                        state.accumulated_tokens.append(limit_tokens)
                        state.accumulated_masks.append([0] * len(limit_tokens))
                        should_continue = True
                    elif state.last_given_index + 1 < len(state.hints):
                        # Select hint (smart or sequential)
                        if self.hint_selector is not None and state.hints_text:
                            # Decode accumulated text for smart selection
                            all_resp_tokens = []
                            for toks in state.accumulated_tokens[1:]:
                                all_resp_tokens.extend(toks)
                            decoded_text = self.tokenizer.decode(all_resp_tokens)

                            hint_text, new_last = self.hint_selector.select_hint_sync(
                                decoded_text, state.hints_text, state.last_given_index,
                            )
                            if hint_text is not None:
                                state.last_given_index = new_last
                                # Tokenize the selected hint text
                                hint_response = f"\n<response>{hint_text}</response>\n<think>\n"
                                hint_tokens = list(self.tokenizer(hint_response, add_special_tokens=False)['input_ids'])
                            else:
                                # Fallback to sequential
                                next_idx = state.last_given_index + 1
                                hint_tokens = state.hints[next_idx]
                                state.last_given_index = next_idx
                        else:
                            # Sequential: use pre-tokenized hints
                            next_idx = state.last_given_index + 1
                            hint_tokens = state.hints[next_idx]
                            state.last_given_index = next_idx

                        state.accumulated_tokens.append(hint_tokens)
                        state.accumulated_masks.append([0] * len(hint_tokens))
                        should_continue = True
                    else:
                        # No more hints - inject message but let model continue to produce an answer
                        no_hints_tokens = list(self.tokenizer(
                            "\n<response>No more hints available.</response>\n",
                            add_special_tokens=False
                        )['input_ids'])
                        state.accumulated_tokens.append(no_hints_tokens)
                        state.accumulated_masks.append([0] * len(no_hints_tokens))
                        should_continue = True

                elif answer_matches:
                    state.completed = True
                else:
                    # No hint or answer - add correction and continue
                    if (state.accumulated_tokens[-1] and
                        state.accumulated_tokens[-1][-1] == self.tokenizer.eos_token_id):
                        state.accumulated_tokens[-1] = state.accumulated_tokens[-1][:-1]
                        state.accumulated_masks[-1] = state.accumulated_masks[-1][:-1]

                    correction_tokens = list(
                        self.tokenizer("\nLet me try again.\n", add_special_tokens=False)['input_ids']
                    )
                    state.accumulated_tokens.append(correction_tokens)
                    state.accumulated_masks.append([0] * len(correction_tokens))
                    should_continue = True

                # Check length limit
                total_len = sum(len(t) for t in state.accumulated_tokens)
                if total_len >= self.config.prompt_length + self.config.response_length:
                    state.completed = True
                    should_continue = False

                # Check turn limit
                state.turn += 1
                if state.turn > self.max_turns:
                    state.completed = True
                    should_continue = False

                if state.completed:
                    completed_count += 1
                    if pbar is not None:
                        pbar.update(1)

                # Re-queue if should continue - use TokensPrompt with flattened tokens
                if should_continue and not state.completed:
                    new_request_id = f"req_{next_request_id}"
                    next_request_id += 1
                    request_to_prompt[new_request_id] = prompt_idx

                    # Flatten all accumulated tokens for the prompt
                    all_tokens = []
                    for token_list in state.accumulated_tokens:
                        all_tokens.extend(token_list)

                    llm_engine.add_request(
                        request_id=new_request_id,
                        prompt=TokensPrompt(prompt_token_ids=all_tokens),
                        params=sampling_params,
                    )

        # Close progress bar
        if pbar is not None:
            pbar.close()

        # Log summary
        total_hints = sum(states[i].num_hints_used for i in range(num_outputs))
        avg_hints = total_hints / num_outputs if num_outputs > 0 else 0
        print(f"[Async Agentic] Completed {num_outputs} prompts, avg hints: {avg_hints:.2f}")

        # Build results
        response_list = []
        mask_list = []
        num_hints_list = []

        for i in range(num_outputs):
            state = states[i]
            # Flatten tokens (skip prompt which is first element)
            response_tokens = []
            response_mask = []
            for tokens, masks in zip(state.accumulated_tokens[1:], state.accumulated_masks):
                response_tokens.extend(tokens)
                response_mask.extend(masks)

            # Truncate to response_length
            response_tokens = response_tokens[:self.config.response_length]
            response_mask = response_mask[:self.config.response_length]

            response_list.append(response_tokens)
            mask_list.append(response_mask)
            num_hints_list.append(state.num_hints_used)

        # Build output tensors
        device = idx.device
        response = pad_2d_list_to_length(
            response_list, self.pad_token_id, max_length=self.config.response_length
        ).to(device)
        document_mask = pad_2d_list_to_length(
            mask_list, 0, max_length=self.config.response_length
        ).to(device)

        # Expand idx, attention_mask, position_ids to match n_samples
        idx = _repeat_interleave(idx, n_samples)
        attention_mask = _repeat_interleave(attention_mask, n_samples)
        position_ids = _repeat_interleave(position_ids, n_samples)
        batch_size = batch_size * n_samples

        # Expand non_tensor_batch
        for key in ["raw_prompt_ids", "target", "hint_exprs"]:
            if key in non_tensor_batch:
                non_tensor_batch[key] = _repeat_interleave(non_tensor_batch[key], n_samples)
        if "numbers" in non_tensor_batch:
            non_tensor_batch["numbers"] = [n for n in non_tensor_batch["numbers"] for _ in range(n_samples)]

        if "epsilon" in non_tensor_batch:
            non_tensor_batch["epsilon"] = _repeat_interleave(non_tensor_batch["epsilon"], n_samples)

        # Build sequence
        seq = torch.cat([idx, response], dim=-1)

        # Build position ids for response
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        # Build attention mask
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        attention_mask_with_documents = torch.cat([attention_mask, document_mask], dim=-1)

        batch_dict = {
            "prompts": idx,
            "responses": response,
            "input_ids": seq,
            "attention_mask": full_attention_mask,
            "document_mask": attention_mask_with_documents,
            "position_ids": position_ids,
            "num_hints": torch.tensor(num_hints_list, dtype=torch.float32),
        }

        return DataProto.from_dict(batch_dict, non_tensors=non_tensor_batch)

    def _simple_generate(self, prompts: DataProto, **kwargs) -> DataProto:
        """Simple generation without multi-turn hints - uses standard LLM.generate()."""
        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch

        do_sample = prompts.meta_info.get("do_sample", True)
        if not do_sample:
            sampling_params = SamplingParams(
                best_of=1, top_p=1.0, top_k=-1, min_p=0.0, temperature=0, n=1,
                max_tokens=self.config.response_length,
                stop=["</answer>", "</think>\n\n<abstain>"],
                include_stop_str_in_output=True,
            )
        else:
            sampling_params = SamplingParams(**self.sampling_params_kwargs)

        # Expand by n_samples
        n_samples = self.n
        raw_prompt_ids = _repeat_interleave(non_tensor_batch["raw_prompt_ids"], n_samples)
        num_outputs = len(raw_prompt_ids)

        # Decode prompts
        prompt_texts = []
        for i in range(num_outputs):
            prompt_ids = raw_prompt_ids[i]
            if isinstance(prompt_ids, np.ndarray):
                prompt_ids = prompt_ids.tolist()
            prompt_texts.append(self.tokenizer.decode(prompt_ids))

        # Generate using standard LLM.generate()
        outputs = self.inference_engine.generate(prompt_texts, sampling_params)

        # Extract results
        results = []
        for output in outputs:
            if output.outputs:
                results.append(list(output.outputs[0].token_ids))
            else:
                results.append([])

        # Build output tensors
        device = idx.device
        response = pad_2d_list_to_length(
            results, self.pad_token_id, max_length=self.config.response_length
        ).to(device)

        idx = _repeat_interleave(idx, n_samples)
        attention_mask = _repeat_interleave(attention_mask, n_samples)
        position_ids = _repeat_interleave(position_ids, n_samples)
        batch_size = batch_size * n_samples

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
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
        }

        return DataProto.from_dict(batch_dict, non_tensors=non_tensor_batch)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)