"""
Example: Custom vllm_speculative Model in lm-evaluation-harness

Usage:
  1) Put this file in lm_eval/models/vllm_speculative.py
  2) Run the harness with:
     lm_eval --model vllm_speculative \\
         --model_args "pretrained=any-string,you-like,tokenizer=Qwen/Qwen2.5-32B-Instruct,speculative_reasoner_host=127.0.0.1,speculative_reasoner_port=5000,thinking_n_ignore=2,drafting_n=1" \\
         --tasks some_task \\
         --batch_size auto \\
         --output_path out.json \\
         --gen_kwargs max_gen_toks=512 \\
         --apply_chat_template
"""

import copy
import requests
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import os
import time
import subprocess
from importlib.util import find_spec

from tqdm import tqdm
from more_itertools import distribute

from lm_eval.api.model import TemplateLM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator, configure_pad_token, undistribute
from lm_eval.utils import (
    eval_logger,
    get_rolling_token_windows,
    make_disjoint_window,
)

try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    pass

def parse_value(s: str):
    """
    Attempt to parse a string as bool -> int -> float -> string.
    """
    # 1) Check for booleans
    lowered = s.lower().strip()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s

def coerce_all_types(param_dict: dict) -> dict:
    out = {}
    for key, val in param_dict.items():
        if isinstance(val, str):
            out[key] = parse_value(val)
        else:
            out[key] = val
    return out

class FakeGenerateOutput:
    def __init__(self, text: str):
        self.outputs = [type("FakeSingleOutput", (object,), {"text": text})()]



class FakeLogprobOutput:
    def __init__(self, n_tokens: int):
        dummy_logprobs = []
        for i in range(n_tokens):
            if i == 0:
                dummy_logprobs.append(None)
            else:
                dummy_logprobs.append({"[FAKE]": 0.0})
        self.prompt_logprobs = dummy_logprobs


@register_model("vllm_speculative")
class SpeculativeVLLM(TemplateLM):
    _DEFAULT_MAX_LENGTH = 22528 # Slightly below 32k

    def __init__(
        self,
        # We keep the same signature as the original VLLM for compatibility,
        # but ignore local model usage. We'll store relevant items for the
        # remote service calls.
        pretrained: str = "meta-llama/Llama-2-7b-chat-hf",
        tokenizer: Optional[str] = None,
        tokenizer_mode: Literal["auto", "slow"] = "auto",
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        max_gen_toks: int = 256,
        max_length: int = None,
        max_model_len: int = None,
        # Some standard harness arguments we won't use for a local model:
        dtype: Literal["float16", "bfloat16", "float32", "auto"] = "auto",
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
        batch_size: Union[str, int] = 1,
        # ... other local vLLM or quantization args omitted for brevity ...
        data_parallel_size: int = 1,
        device: str = "cuda",
        # We'll accept our "remote" parameters:
        speculative_reasoner_host: str = "127.0.0.1",
        speculative_reasoner_port: int = 5000,
        # Additional reasoner config (like # of steps) can come in via kwargs:
        **kwargs,
    ):
        """
        The harness will call this constructor with --model_args. For instance:
          --model_args "pretrained=...,tokenizer=...,speculative_reasoner_host=...,thinking_n_ignore=2,..."
        We'll store the remote host/port, plus any additional args in `self.service_params`.
        """
        super().__init__()

        kill_cmd = "fuser -k -9 /dev/nvidia*"
        subprocess.run(kill_cmd, shell=True)
        self.service_host = speculative_reasoner_host
        self.service_port = speculative_reasoner_port
        self.service_params = coerce_all_types(kwargs)

        # after that, read out each new param, with defaults, e.g.:
        self.placeholder_mode = self.service_params.get("placeholder_mode", False)
        self.spec_rewrite = self.service_params.get("spec_rewrite", False)
        self.logprob_subselect = self.service_params.get("logprob_subselect", False)
        self.big_model_only = self.service_params.get("big_model_only", False)
        self.small_model_only = self.service_params.get("small_model_only", False)

        # likewise for numeric fields:
        self.sgen = self.service_params.get("sgen", 256)
        self.stok = self.service_params.get("stok", 16)
        self.sdecay = self.service_params.get("sdecay", 2)
        self.ltok = self.service_params.get("ltok", 16)
        self.lbound = self.service_params.get("lbound", 4)


        self.service_script_path = self.service_params.get("service_script_path", "./spec_service.py")
        self._ensure_correct_service_is_running()
        big_model_name = self.service_params.get("big_model", pretrained)

        if max_length and max_model_len:
            raise ValueError("Either max_length or max_model_len may be provided, not both.")
        self._max_length = max_model_len if max_model_len is not None else max_length
        if not self._max_length:
            self._max_length = self._DEFAULT_MAX_LENGTH

        self._max_gen_toks = max_gen_toks
        self.data_parallel_size = data_parallel_size
        self.add_bos_token = add_bos_token
        self.custom_prefix_token_id = prefix_token_id

        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else batch_size
        )
        if not find_spec("vllm"):
            raise RuntimeError("`vllm` is not installed.")
        from vllm.transformers_utils.tokenizer import get_tokenizer
        self.tokenizer = get_tokenizer(
            big_model_name,
            tokenizer_mode="auto",  # or from your constructor arguments
            trust_remote_code=True   # or from your constructor args
        )
        self.tokenizer = configure_pad_token(self.tokenizer)

        # For example, some folks want to forcibly add a BOS token
        if add_bos_token:
            # We'll rely on the harness code to handle that in tok_encode
            pass

        eval_logger.info(
            f"[SpeculativeVLLM] Remote generation from service at "
            f"http://{self.service_host}:{self.service_port}. "
            f"Max length: {self._max_length}, default max_gen_toks: {self._max_gen_toks}."
        )

    def _ensure_correct_service_is_running(self):
        ping_url = f"http://{self.service_host}:{self.service_port}/ping"
        if self._ping_service(ping_url):
            eval_logger.info(f"[SpeculativeVLLM] Service is already running; attempting shutdown.")
            
            # Attempt a graceful shutdown via /shutdown
            shutdown_url = f"http://{self.service_host}:{self.service_port}/shutdown"
            try:
                requests.post(shutdown_url, timeout=5)
            except requests.exceptions.RequestException:
                pass
            
            # Wait for it to actually stop responding
            start_time = time.time()
            while self._ping_service(ping_url):
                if time.time() - start_time > 30:
                    raise RuntimeError("[SpeculativeVLLM] Could not shut down the service within 30s; exiting.")
                time.sleep(1)
            eval_logger.info("[SpeculativeVLLM] Old service has been shut down.")
        
        # Now start the service fresh with updated arguments
        eval_logger.info("[SpeculativeVLLM] Launching spec_service.py...")

        # Gather the arguments that your spec_service.py expects:
        big_model = self.service_params.get("big_model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        big_model_port = self.service_params.get("big_model_port", 8000)
        big_model_gpus = str(self.service_params.get("big_model_gpus", "0,1")).replace("|", ",")
        small_model = self.service_params.get("small_model", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
        small_model_port = self.service_params.get("small_model_port", 8001)
        small_model_gpus = str(self.service_params.get("small_model_gpus", "2")).replace("|", ",")
        
        thinking_n_ignore = self.service_params.get("thinking_n_ignore", 2)
        drafting_n = self.service_params.get("drafting_n", 1)
        bloat_tokens = self.service_params.get("bloat_tokens", 0)
        max_tokens = self.service_params.get("max_tokens", 16384)
        terminating_string = self.service_params.get(
            "terminating_string",
            r"\nPut your final answer within \boxed{}."
        )

        cmd = [
            "python", self.service_script_path,
            f"--big_model={big_model}",
            f"--big_model_port={big_model_port}",
            f"--big_model_gpus={big_model_gpus}",
            f"--small_model={small_model}",
            f"--small_model_port={small_model_port}",
            f"--small_model_gpus={small_model_gpus}",
            f"--thinking_n_ignore={thinking_n_ignore}",
            f"--drafting_n={drafting_n}",
            f"--bloat_tokens={bloat_tokens}",
            f"--max_tokens={max_tokens}",
            f"--terminating_string={terminating_string}",
            f"--sgen={self.sgen}",
            f"--stok={self.stok}",
            f"--sdecay={self.sdecay}",
            f"--ltok={self.ltok}",
            f"--lbound={self.lbound}",
            "--port", str(self.service_port),
        ]

        if self.service_params.get("small_first", False):
            cmd.append("--small_first")
        if self.service_params.get("spec_rewrite", False):
            cmd.append("--spec_rewrite")
        if self.service_params.get("logprob_subselect", False):
            cmd.append("--logprob_subselect")
        if self.service_params.get("big_model_only", False):
            cmd.append("--big_model_only")
        if self.service_params.get("small_model_only", False):
            cmd.append("--small_model_only")
        if self.service_params.get("full_rewrite", False):
            cmd.append("--full_rewrite")
        if self.service_params.get("draft_propose_ignore_str", False):
            cmd.append("--draft_propose_ignore_str")
        if self.service_params.get("terminate_on_exit", False):
            cmd.append("--terminate_on_exit")
        if self.service_params.get("test_logging", False):
            cmd.append("--test_logging")

        eval_logger.info(f"[SpeculativeVLLM] Launching: {' '.join(cmd)}")
        env = os.environ.copy()
        self.service_proc = subprocess.Popen(cmd, env=env)

        # Wait until the service is up
        start_time = time.time()
        timeout = 600
        while time.time() - start_time < timeout:
            if self._ping_service(ping_url):
                eval_logger.info("[SpeculativeVLLM] Service started successfully.")
                return
            if self.service_proc.poll() is not None:
                # The process terminated unexpectedly
                raise RuntimeError("[SpeculativeVLLM] Service process exited prematurely.")
            time.sleep(2)

        raise RuntimeError("[SpeculativeVLLM] Service did not respond within 600 seconds.")


    def _ping_service(self, url: str) -> bool:
        """
        Return True if GET /ping returns 200, else False.
        """
        try:
            r = requests.get(url, timeout=3.0)
            return (r.status_code == 200)
        except Exception:
            return False


    @property
    def eot_token_id(self) -> int:
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def prefix_token_id(self) -> int:
        # Used as prefix for loglikelihood.
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=True
        )

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        if not add_special_tokens:
            add_special_tokens = False or self.add_bos_token
        encoding = self.tokenizer(
            string,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            return_attention_mask=False,
        ).input_ids

        if left_truncate_len:
            if not isinstance(string, str):
                encoding = [enc[-left_truncate_len:] for enc in encoding]
            else:
                encoding = encoding[-left_truncate_len:]

        return encoding

    def _model_generate(
        self,
        encoded_prompts: List[List[int]],
        generate: bool = False,
        max_tokens: int = None,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        The harness calls this inside `generate_until` or `_loglikelihood_tokens`.
        We replicate the shape of the result, but we skip local vLLM usage.

        - If generate=True, we do a remote call to your speculation endpoint.
        - If generate=False, this is the log-likelihood scenario. We'll just
          produce dummy outputs with fake logprobs (unless you have an endpoint
          for real log-likelihood).
        """

        if generate:
            completions = []
            req_idx = 0
            for token_ids in tqdm(encoded_prompts, desc="Sending service requests"):
                prompt_text = self.tokenizer.decode(token_ids)
                print(f"\n\n Request {req_idx}:\n{prompt_text}\n\n")
                req_idx += 1
                payload = {
                    "question": prompt_text,
                    "max_tokens": max_tokens if max_tokens else self._max_gen_toks,
                }
                if stop:
                    payload["terminating_string"] = stop[-1]

                payload.update(self.service_params)
                payload.update(kwargs)

                url = f"http://{self.service_host}:{self.service_port}/speculative_reason"
                resp = requests.post(url, json=payload)
                resp.raise_for_status()
                final_answer = resp.json().get("final_answer", "")

                completions.append(FakeGenerateOutput(final_answer))

            return completions

        else:
            logprob_outputs = []
            for token_ids in encoded_prompts:
                n_tokens = len(token_ids)
                logprob_outputs.append(FakeLogprobOutput(n_tokens))

            return logprob_outputs

    def loglikelihood_rolling(self, requests: List[Instance], disable_tqdm: bool = False) -> List[float]:
        """
        For tasks that measure perplexity or log-likelihood over sliding windows,
        we replicate the original code's logic with get_rolling_token_windows,
        but we produce dummy log-likelihood values (0.0) since we can't do it remotely
        unless your service supports it.
        """
        loglikelihoods = []
        for (string,) in tqdm([req.args for req in requests], disable=disable_tqdm):
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length - 1,
                        context_len=1,
                    ),
                )
            )
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll_list = self._loglikelihood_tokens(rolling_token_windows)
            total_ll = sum([x[0] for x in string_nll_list])
            loglikelihoods.append(total_ll)

            self.cache_hook.add_partial("loglikelihood_rolling", (string,), total_ll)

        return loglikelihoods

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False) -> List[str]:
        """
        For generation tasks. We'll replicate the original chunking logic, but
        internally, we call _model_generate with `generate=True`, which triggers
        the remote request.
        """
        res = []

        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding: List[List[int]] = self.tok_encode(context, add_special_tokens=self.add_bos_token)
        requests_ = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            return -len(_requests[0][1]), _requests[0][0]

        re_ords = Collator(requests_, _collate_gen, group_by="gen_kwargs")
        chunks = re_ords.get_batched(
            n=0 if self.batch_size == "auto" else int(self.batch_size),
            batch_fn=None
        )

        pbar = tqdm(
            total=len(requests_),
            disable=disable_tqdm,
            desc="Running generate_until requests",
        )

        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context_list, context_enc_list = zip(*context_and_encoding)

            gen_kwargs = all_gen_kwargs[0]
            if isinstance(gen_kwargs, dict):
                kwargs = copy.deepcopy(gen_kwargs)
            else:
                raise ValueError(f"Expected gen_kwargs dict but got: {gen_kwargs}")

            until = None
            if "until" in kwargs:
                until = kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `until` to be str or list, but got {until}")
            eos = self.tokenizer.decode(self.eot_token_id)
            if not until:
                until = [eos]
            else:
                until.append(eos)

            if "max_gen_toks" in kwargs:
                max_gen_toks = kwargs.pop("max_gen_toks")
            else:
                max_gen_toks = self._max_gen_toks

            max_ctx_len = self.max_length - max_gen_toks
            context_enc_list = [x[-max_ctx_len:] for x in context_enc_list]

            outputs = self._model_generate(
                encoded_prompts=context_enc_list,
                generate=True,
                max_tokens=max_gen_toks,
                stop=until,
                **kwargs
            )

            for out_obj, original_context in zip(outputs, context_list):
                generated_text = out_obj.outputs[0].text
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (original_context, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """
        The harness calls this to get log-likelihood for each "context + continuation".
        We'll replicate chunking & call _model_generate with generate=False => returns
        FakeLogprobOutput objects. Then parse them with _parse_logprobs.
        """
        res = []

        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=0 if self.batch_size == "auto" else int(self.batch_size),
            batch_fn=None
        )

        pbar = tqdm(total=len(requests), disable=disable_tqdm, desc="Running loglikelihood requests")

        for chunk in chunks:
            inputs = []
            ctxlens = []
            for cache_key, context_enc, continuation_enc in chunk:
                # potentially left-truncate if longer than self.max_length
                joined = (context_enc + continuation_enc)[-self.max_length:]
                # how many tokens for context portion?
                ctxlen = len(context_enc) - max(0, len(context_enc)+len(continuation_enc) - self.max_length)
                inputs.append(joined)
                ctxlens.append(ctxlen)

            outputs = self._model_generate(requests=inputs, generate=False)
            # outputs is a list of FakeLogprobOutput objects

            for output, ctxlen, (cache_key, _, _), inp in zip(outputs, ctxlens, chunk, inputs):
                answer = self._parse_logprobs(tokens=inp, outputs=output, ctxlen=ctxlen)
                res.append(answer)

                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    @staticmethod
    def _parse_logprobs(tokens: List[int], outputs: FakeLogprobOutput, ctxlen: int) -> Tuple[float, bool]:
        """
        In the real vLLM, we sum the logprobs of each continuation token and
        check if it was the top choice. We have a dummy logprob distribution,
        so let's just return (0.0, True).
        """
        return 0.0, True

    @staticmethod
    def modify_gen_kwargs(kwargs: dict) -> dict:
        """
        The original vllm code does some transformations if do_sample=False
        or if temperature not set. We can replicate or skip.
        """
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False and "temperature" not in kwargs:
            eval_logger.debug("Got do_sample=False with no temperature => set temp=0.0")
            kwargs["temperature"] = 0.0
        kwargs["skip_special_tokens"] = kwargs.get("skip_special_tokens", False)
        kwargs["spaces_between_special_tokens"] = kwargs.get("spaces_between_special_tokens", False)
        return kwargs
