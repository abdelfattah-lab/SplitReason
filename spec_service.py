import argparse
import os
import subprocess
import time
import signal
import sys
import requests
import threading

# from flask import Flask, request, jsonify
import base64, numpy as np, torch

import sys
from pathlib import Path
from flask import Flask, request, jsonify
import torch, uuid, json
from vllm import LLM, SamplingParams
from typing import List, Tuple, Dict, Any, Optional, Union
import re
import collections

# from modes.speculative_reasoning_perf_profiling import run_speculative_reasoning_flow_perf_only
from modes.spec_rewrite import run_speculative_rewrite_flow
from modes.speculative_reasoning import run_speculative_reasoning_flow
from modes.speculative_reasoning_perf import run_speculative_reasoning_flow_perf
from modes.placeholder import run_placeholder_flow
from modes.logprob_subselect import run_logprob_subselect_flow
from modes.small_model_only import run_smallmodel_flow
from modes.big_model_only import run_bigmodel_flow
from modes.random_switch_flow import run_random_switch_flow
from modes.speculative_decoding import run_speculative_decoding_flow
from transformers import AutoTokenizer

import json
big_model_tokenizer = None

from pathlib import Path
env = os.environ.copy()

# add the directory that contains weight_sync_ext/ to PYTHONPATH
project_root = Path(__file__).resolve().parent
env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"


app = Flask(__name__)
SMALL_LLM          = None   # vllm.LLM object (small model)
SMALL_MODEL_RUNNER = None   # its ModelRunner – we need it for load_weights()
big_model_proc = None
small_model_proc = None
service_args = None   # Will hold the parsed arguments

def wait_for_vllm(port: int, timeout: float = 600.0, path: str = "/health") -> bool:
    """Poll the given vLLM server until it returns HTTP 200."""
    url = f"http://localhost:{port}{path}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url, timeout=2).status_code == 200:   # ≤2 s socket timeout
                return True
        except requests.RequestException:
            pass
        time.sleep(1.0)  # 1 Hz polling is fine; vLLM boots in 10-120 s
    return False

def wait_for_vllm_ready(port: int, model_id, timeout: float = 600.0) -> bool:
    """
    Wait until the vLLM HTTP server is up **and** has finished loading at
    least one model (checked via /v1/models).
    """
    chat = True
    url = f"http://localhost:{port}/v1/" + ("chat/completions" if chat else "completions")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 6,
        "temperature": 0,
        "stream": False,
    }

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.post(url, json=payload, timeout=5)
            r.raise_for_status()          # catch 4xx / 5xx early
            try:
                data = r.json()
            except (ValueError, json.JSONDecodeError):
                raise RuntimeError("Non-JSON response")

            if "choices" in data:
                print(f"[Service] vLLM is ready on {url}")
                return True
        except Exception as e:             # broad: network or decode
            print(f"[Service] still waiting: {e}")

        time.sleep(2)                      # calmer log
    return False


def wait_for_server(url, timeout=600.0):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

def load_big_model_tokenizer(model_name: str):
    """
    Load and cache the big model's tokenizer once.
    """
    global big_model_tokenizer
    if big_model_tokenizer is None:
        big_model_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return big_model_tokenizer

def approximate_token_count(text: str) -> int:
    """
    Returns the approximate number of tokens in `text` using the big model's tokenizer.
    """
    tokens = big_model_tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def _start_vllm(model_name: str, port: int, gpu_ids: str, is_small: bool) -> subprocess.Popen:

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    tp_size = len(gpu_ids.split(","))
    if is_small:
        assert tp_size == 1, "Small model must run on a single GPU"
        # "--max-model-len", "8192",
        # "--max-model-len", "32768",
        # "--max-model-len", "16384",
    project_root = Path(__file__).resolve().parent  # directory containing weight_sync_ext
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"
    vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")  # …/envs/diffusedreasoner/bin/vllm

    # cmd = [sys.executable, "-m", 
        # "vllm",
    cmd = [vllm_bin, "serve", model_name,
        "--port", str(port),
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", "16384",
        "--uvicorn-log-level=warning",
        "--enable-prefix-caching",
        "--enable-chunked-prefill"
    ]
    # item_small = ["--worker-extension-cls", "weight_sync_ext.weight_sync_worker.WorkerExtension",]
    # if is_small:
        # cmd.extend(item_small)
    return subprocess.Popen(cmd, env=env)

    # item_small = ["--worker-extension-cls", "vllm.examples.offline_inference.rlhf_utils.WorkerExtension",]

def _launch_blocking(model_name: str, port: int, gpu_ids: str,
                     timeout: float = 600.0, poll: float = 5.0, is_small=False) -> subprocess.Popen:
    """Start vLLM and block until it answers a ping or timeout expires."""
    proc = _start_vllm(model_name, port, gpu_ids, is_small=is_small)
    print(f"[Service] Launching {model_name} on :{port} (GPUs={gpu_ids}) …")

    elapsed = 0.0
    while elapsed < timeout:
        if wait_for_vllm_ready(port, model_name, timeout=poll):
            print(f"[Service] {model_name} on :{port} is ready! ✅")
            return proc
        elapsed += poll
        print(f"[Service] ... still waiting for {model_name} on :{port}\t({int(elapsed)} s/{int(timeout)} s)")

    proc.terminate()
    proc.wait()
    raise RuntimeError(
        f"[Service] {model_name} on :{port} did not become ready within {timeout} s"
    )


def launch_big_model_vllm(big_model, port, gpu_ids):
    return _launch_blocking(big_model, port, gpu_ids)


def launch_small_model(model_name, port, gpu_ids):
    return _launch_blocking(model_name, port, gpu_ids, is_small=True)

def launch_spec_decoding_server(
    big_model: str,
    port: int,
    gpu_ids: str,
    seed: int,
    gpu_memory_utilization: float,
    speculative_config: str,
    max_model_len: int,
    enforce_eager: bool,
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    tp_size = len(gpu_ids.split(","))

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--model", big_model,
        "--seed", str(seed),
        "--tensor-parallel-size", str(tp_size),
        "--gpu_memory_utilization", str(gpu_memory_utilization),
        "--speculative_config", speculative_config,
        "--max-model-len", str(max_model_len),
        "--enable_prefix_caching",
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")

    print(f"[Service] Launching speculative‑decoding server on port {port} with model {big_model}")
    return subprocess.Popen(cmd, env=env)


def batched_eval_logprob_vllm(
    text_batch: List[str],
    big_model_port: int,
    big_model: str,
    requests,
    temperature: float = 0.0,
    max_tokens: int = 0,
    is_bigmodel_halting=False,
    logprobs: int = 1
) -> Tuple[List[float], List[str], float, int]:
    """
    Evaluate the total log-likelihood of each string in `text_batch` 
    by calling vLLM in a *single* batch request. Returns:
      (scores, generated_texts, avg_latency, num_requests).

    - scores: total log-likelihood (or average log-likelihood, per your usage).
    - generated_texts: text generated by the model for each prompt.
    - avg_latency: average request latency for the entire batch.
    - num_requests: length of text_batch (the number of prompts in the batch).
    """
    url = f"http://localhost:{big_model_port}/v1/completions"
    if is_bigmodel_halting:
        payload = {
            "model": big_model,
            "prompt": text_batch,
            "max_tokens": max_tokens,   # typically 0 or 1 if purely evaluating logprobs
            "temperature": temperature,
            "logprobs": logprobs,
            "stop": ["<bigmodel>"],
            "include_stop_str_in_output": True,
            "n": 1,                     # 1 completion per prompt
        }
    else:
        payload = {
            "model": big_model,
            "prompt": text_batch,
            "max_tokens": max_tokens,   # typically 0 or 1 if purely evaluating logprobs
            "temperature": temperature,
            "logprobs": logprobs,
            "n": 1,                     # 1 completion per prompt
        }

    start_time = time.time()
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    end_time = time.time()
    resp_json = resp.json()

    choices = resp_json["choices"]

    scores = []
    gentexts = []

    # We'll parse each choice in order. Each item in `choices`
    # corresponds to the matching prompt in `text_batch`.
    for i, choice in enumerate(choices):
        generated_text = choice["text"]
        logprob_info = choice.get("logprobs", {})
        token_logprobs = logprob_info.get("token_logprobs", [])

        # Filter out None
        token_logprobs = [p for p in token_logprobs if p is not None]

        if len(token_logprobs) > 0:
            # As in your existing code, you used the average of token_logprobs as "score"
            prompt_ll = sum(token_logprobs) / len(token_logprobs)
        else:
            prompt_ll = float("-inf")  # or 0.0, depending on your desired fallback

        scores.append(prompt_ll)
        gentexts.append(generated_text)

    total_latency = end_time - start_time
    avg_latency = total_latency / max(1, len(text_batch))
    num_requests = len(text_batch)

    return scores, gentexts, avg_latency, num_requests

def _get_spec_metrics(port: int, model_name: str) -> dict:
    """
    Scrape vLLM's /metrics endpoint and parse out the five
    speculative‑decoding values into a dict.
    """
    text = requests.get(f"http://localhost:{port}/metrics").text
    patterns = {
        "accepted":        "spec_decode_num_accepted_tokens_total",
        "draft":           "spec_decode_num_draft_tokens_total",
        "emitted":         "spec_decode_num_emitted_tokens_total",
        "acceptance_rate": "spec_decode_draft_acceptance_rate",
        "efficiency":      "spec_decode_efficiency",
    }
    out = {}
    for key, prom in patterns.items():
        m = re.search(
            rf'vllm:{prom}\{{model_name="{model_name}"\}} ([0-9.]+)',
            text
        )
        out[key] = float(m.group(1)) if m else 0.0
    return out


# ------------------------------------------------------------
# Local helpers for OpenAI-style output from vllm.LLM.generate
# ------------------------------------------------------------
def _to_openai_single(out, want_tokens=False):
    """Convert one vLLM SequenceOutput to the OpenAI /v1/completions shape."""
    choice = {
        "index": 0,
        "text":  out.text,
        "finish_reason": getattr(out, "finish_reason", None),
    }
    print("[Service] finish_reason:", choice["finish_reason"])
    if want_tokens:
        choice["logprobs"] = {
            "tokens":            out.tokens,
            "token_logprobs":    out.logprobs,
            "top_logprobs":      None,
            "text_offset":       None,
        }
    return {"choices": [choice]}


# ------------------------------------------------------------
# SINGLE prompt ­– generate_text_vllm
# ------------------------------------------------------------
def generate_text_vllm(prompt,
                       port          = 8000,
                       temperature   = 0.6,
                       max_tokens    = 128,
                       model         = "my-model",
                       is_bigmodel_halting=False,
                       speculative_decoding=False):
    """
    Exactly the same signature as before.
    Falls back to HTTP unless the request is for the embedded small model.
    """
    # -------- local fast path ---------------------------------
    if SMALL_LLM is not None and port == service_args.small_model_port:
        stop = ["<bigmodel>"] if is_bigmodel_halting else None
        params = SamplingParams(
            temperature = temperature,
            max_tokens  = max_tokens,
            stop        = stop,
            include_stop_str_in_output = True,
        )
        t0 = time.time()
        out = SMALL_LLM.generate([prompt], params)[0].outputs[0]
        latency = time.time() - t0
        return _to_openai_single(out), latency

    # -------- original HTTP path ------------------------------
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model":       model,
        "prompt":      prompt,
        "max_tokens":  max_tokens,
        "temperature": temperature,
    }
    if is_bigmodel_halting:
        payload["stop"] = ["<bigmodel>"]
        payload["include_stop_str_in_output"] = True

    if speculative_decoding:
        before = _get_spec_metrics(port, model)

    t0   = time.time()
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    latency = time.time() - t0

    if speculative_decoding:
        after   = _get_spec_metrics(port, model)
        metrics = {
            "accepted_tokens": after["accepted"] - before["accepted"],
            "draft_tokens":    after["draft"]    - before["draft"],
            "emitted_tokens":  after["emitted"]  - before["emitted"],
            "acceptance_rate": after["acceptance_rate"],
            "efficiency":      after["efficiency"],
        }
        return resp.json(), latency, metrics

    return resp.json(), latency


# ------------------------------------------------------------
# BATCH ­– batched_generate_text_vllm
# ------------------------------------------------------------
def batched_generate_text_vllm(prompts: List[str],
                               port: int              = 8000,
                               temperature: float     = 0.6,
                               max_tokens: int        = 128,
                               model: str             = "my-model",
                               is_bigmodel_halting    = False,
                               requests               = None) -> Tuple[List[dict], float]:

    # -------- local fast path ---------------------------------
    if SMALL_LLM is not None and port == service_args.small_model_port:
        stop = ["<bigmodel>"] if is_bigmodel_halting else None
        params = SamplingParams(
            temperature = temperature,
            max_tokens  = max_tokens,
            n           = 1,
            stop        = stop,
            include_stop_str_in_output = True,
        )
        t0   = time.time()
        outs = SMALL_LLM.generate(prompts, params)
        latency = (time.time() - t0) / max(1, len(prompts))
        results = [_to_openai_single(o.outputs[0]) for o in outs]
        return results, latency

    # -------- original HTTP path ------------------------------
    if requests is None:
        import requests as _requests
        requests = _requests

    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model":       model,
        "prompt":      prompts,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "n":           1,
    }
    if is_bigmodel_halting:
        payload.update({
            "stop": ["<bigmodel>"],
            "include_stop_str_in_output": True,
        })

    t0   = time.time()
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    latency = (time.time() - t0) / max(1, len(prompts))

    choices = resp.json()["choices"]
    results = [{"choices": [c]} for c in choices]
    return results, latency


# ------------------------------------------------------------
# BATCH + TOKENS ­– batched_generate_text_with_tokens_vllm
# ------------------------------------------------------------
def batched_generate_text_with_tokens_vllm(prompts: List[str],
                                           port: int              = 8000,
                                           temperature: float     = 0.6,
                                           max_tokens: int        = 128,
                                           model: str             = "my-model",
                                           requests               = None,
                                           is_bigmodel_halting    = False,
                                           logprobs: int          = 1
                                           ) -> Tuple[List[dict], List[List[str]], float]:

    # -------- local fast path ---------------------------------
    if SMALL_LLM is not None and port == service_args.small_model_port:
        stop = ["<bigmodel>"] if is_bigmodel_halting else None
        params = SamplingParams(
            temperature = temperature,
            max_tokens  = max_tokens,
            n           = 1,
            logprobs    = logprobs,
            stop        = stop,
            include_stop_str_in_output = True,
        )
        t0   = time.time()
        outs = SMALL_LLM.generate(prompts, params)
        latency = (time.time() - t0) / max(1, len(prompts))

        results     = []
        tokens_list = []
        tok = SMALL_LLM.get_tokenizer()      # available on recent vLLM
        for o in outs:
            seq = o.outputs[0]
            results.append(_to_openai_single(seq, want_tokens=True))
            tokens_list.append(tok.convert_ids_to_tokens(seq.token_ids))

        return results, tokens_list, latency

    # -------- original HTTP path ------------------------------
    if requests is None:
        import requests as _requests
        requests = _requests

    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model":       model,
        "prompt":      prompts,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "logprobs":    logprobs,
        "n":           1,
    }
    if is_bigmodel_halting:
        payload.update({
            "stop": ["<bigmodel>"],
            "include_stop_str_in_output": True,
        })

    t0   = time.time()
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    latency = (time.time() - t0) / max(1, len(prompts))

    choices = resp.json()["choices"]
    results = [{"choices": [c]} for c in choices]
    tokens  = [c.get("logprobs", {}).get("tokens", []) for c in choices]
    return results, tokens, latency


def extract_cot(raw_text, think_suffix, fallback_suffixes=()):
    """
    Attempt to split the model's reply to isolate the chain-of-thought portion.
    If think_suffix is present, we split on it. Otherwise try fallback suffixes.
    Otherwise return entire text.
    """
    if think_suffix and think_suffix in raw_text:
        return raw_text.split(think_suffix)[0]

    for fb in fallback_suffixes:
        if fb in raw_text:
            return raw_text.split(fb)[0]

    return raw_text

@app.route("/shutdown", methods=["POST"])
def shutdown():
    def stop_server():
        time.sleep(0.2)
        os._exit(0) 
    threading.Thread(target=stop_server).start()
    return "Shutting down..."



ready_event = threading.Event()

@app.route("/ping", methods=["GET"])
def ping():
    """
    Health-check:
      • 200  → service + models are ready
      • 503  → Flask is up but a model is still loading
    """
    if ready_event.is_set():
        return jsonify({"message": "pong"}), 200
    return jsonify({"message": "warming-up"}), 503

# @app.route("/ping", methods=["GET"])
# def ping():
#     """
#     Simple health check endpoint to verify service is up.
#     """
#     return jsonify({"message": "pong"}), 200

@app.route("/speculative_reason", methods=["POST"])
def speculative_reason():
    data = request.json
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in JSON payload"}), 400

    question = data["question"]
    test_logging = data.get("test_logging", False)

    # Grab all the usual arguments (or use defaults from CLI)
    thinking_n_ignore = data.get("thinking_n_ignore", service_args.thinking_n_ignore)
    drafting_n = data.get("drafting_n", service_args.drafting_n)
    full_rewrite = data.get("full_rewrite", service_args.full_rewrite)
    max_tokens = data.get("max_tokens", service_args.max_tokens)
    temperature = data.get("temperature", 0.6)
    terminating_string = data.get("terminating_string", service_args.terminating_string)
    draft_propose_ignore_str = data.get("draft_propose_ignore_str", service_args.draft_propose_ignore_str)
    small_first = data.get("small_first", service_args.small_first)
    bloat_tokens = data.get("bloat_tokens", service_args.bloat_tokens)
    switch_ratio = data.get("switch_ratio", service_args.switch_ratio)
    switch_chunk = data.get("switch_chunk", service_args.switch_chunk)

    if data.get("spec_reason_perf", False):
        final_reply, usage_data = run_speculative_reasoning_flow_perf(
            question=question,
            sgen=data.get("sgen", service_args.sgen),
            stok=data.get("stok", service_args.stok),
            sdecay=data.get("sdecay", service_args.sdecay),
            ltok=data.get("ltok", service_args.ltok),
            max_tokens=max_tokens,
            temperature=temperature,
            big_model=service_args.big_model,
            big_model_port=service_args.big_model_port,
            small_model=service_args.small_model,
            small_model_port=service_args.small_model_port,
            requests=requests,
            batched_generate_text_vllm=batched_generate_text_vllm,
            batched_generate_text_with_tokens_vllm=batched_generate_text_with_tokens_vllm,
            batched_eval_logprob_vllm=batched_eval_logprob_vllm,
            terminating_string=terminating_string,
            test_logging=test_logging,
            lbound=data.get("lbound", service_args.lbound),
            max_iterations=data.get("max_iterations", service_args.max_iterations),
            sequential_scale=data.get("sequential_scale", service_args.sequential_scale),
            token_counter=approximate_token_count
        )
    elif data.get("placeholder_mode", False):
        final_reply, usage_data = run_placeholder_flow(
            question=question,
            big_model=service_args.big_model,
            big_model_port=service_args.big_model_port,
            small_model=service_args.small_model,
            small_model_port=service_args.small_model_port,
            generate_text_vllm=generate_text_vllm,
            test_logging=test_logging,
            max_tokens=max_tokens,
            temperature=temperature
        )
    elif data.get("small_model_only", False):
        final_reply, usage_data = run_smallmodel_flow(
            question=question,
            small_model=service_args.small_model,
            small_model_port=service_args.small_model_port,
            generate_text_vllm=generate_text_vllm,
            terminating_string=terminating_string,
            max_tokens=max_tokens,
            test_logging=test_logging,
            temperature=temperature,
            sequential_scale=data.get("sequential_scale", service_args.sequential_scale),
            token_counter=approximate_token_count
        )
    elif data.get("big_model_only", False):
        final_reply, usage_data = run_bigmodel_flow(
            question=question,
            big_model=service_args.big_model,
            big_model_port=service_args.big_model_port,
            generate_text_vllm=generate_text_vllm,
            test_logging=test_logging,
            max_tokens=max_tokens,
            terminating_string=terminating_string,
            temperature=temperature,
            sequential_scale=data.get("sequential_scale", service_args.sequential_scale),
            token_counter=approximate_token_count
        )
    elif data.get("logprob_subselect", False):
        final_reply, usage_data = run_logprob_subselect_flow(
            question=question,
            sgen=data.get("sgen", service_args.sgen),
            stok=data.get("stok", service_args.stok),
            sdecay=data.get("sdecay", service_args.sdecay),
            ltok=data.get("ltok", service_args.ltok),
            max_tokens=max_tokens,
            temperature=temperature,
            big_model=service_args.big_model,
            big_model_port=service_args.big_model_port,
            small_model=service_args.small_model,
            small_model_port=service_args.small_model_port,
            requests=requests,
            batched_generate_text_vllm=batched_generate_text_vllm,
            batched_eval_logprob_vllm=batched_eval_logprob_vllm,
            terminating_string=terminating_string,
            test_logging=test_logging,
            lbound=data.get("lbound", service_args.lbound),
            max_iterations=data.get("max_iterations", service_args.max_iterations),
            sequential_scale=data.get("sequential_scale", service_args.sequential_scale)
        )
    elif data.get("spec_rewrite", False):
        final_reply, usage_data = run_speculative_rewrite_flow(
            question=question,
            test_logging=test_logging,
            thinking_n_ignore=thinking_n_ignore,
            drafting_n=drafting_n,
            full_rewrite=full_rewrite,
            temperature=temperature,
            max_tokens=max_tokens,
            terminating_string=terminating_string,
            draft_propose_ignore_str=draft_propose_ignore_str,
            small_first=small_first,
            big_model_port=service_args.big_model_port,
            big_model=service_args.big_model,
            small_model_port=service_args.small_model_port,
            small_model=service_args.small_model,
            bloat_tokens=bloat_tokens,
            generate_text_vllm=generate_text_vllm,
            extract_cot=extract_cot,
            service_args=service_args
        )
    elif data.get("random_switch", False):
        final_reply, usage_data = run_random_switch_flow(
            question=question,
            test_logging=test_logging,
            temperature=temperature,
            max_tokens=max_tokens,
            terminating_string=terminating_string,
            big_model_port=service_args.big_model_port,
            big_model=service_args.big_model,
            small_model_port=service_args.small_model_port,
            small_model=service_args.small_model,
            batched_generate_text_vllm=batched_generate_text_vllm,
            batched_eval_logprob_vllm=batched_eval_logprob_vllm,
            switch_ratio=switch_ratio,
            switch_chunk=switch_chunk,
            requests=requests
        )
    elif data.get("spec_reason", False):
        final_reply, usage_data = run_speculative_reasoning_flow(
            question=question,
            sgen=data.get("sgen", service_args.sgen),
            stok=data.get("stok", service_args.stok),
            sdecay=data.get("sdecay", service_args.sdecay),
            ltok=data.get("ltok", service_args.ltok),
            max_tokens=max_tokens,
            temperature=temperature,
            big_model=service_args.big_model,
            big_model_port=service_args.big_model_port,
            small_model=service_args.small_model,
            small_model_port=service_args.small_model_port,
            requests=requests,
            batched_generate_text_vllm=batched_generate_text_vllm,
            batched_generate_text_with_tokens_vllm=batched_generate_text_with_tokens_vllm,
            batched_eval_logprob_vllm=batched_eval_logprob_vllm,
            terminating_string=terminating_string,
            test_logging=test_logging,
            lbound=data.get("lbound", service_args.lbound),
            max_iterations=data.get("max_iterations", service_args.max_iterations),
            sequential_scale=data.get("sequential_scale", service_args.sequential_scale),
            token_counter=approximate_token_count
        )
    elif data.get("spec_decoding", False):
        final_reply, usage_data = run_speculative_decoding_flow(
            question=question,
            big_model=service_args.big_model,
            big_model_port=service_args.big_model_port,
            generate_text_vllm=generate_text_vllm,
            max_tokens=max_tokens,
            temperature=temperature,
            test_logging=test_logging,
            token_counter=approximate_token_count,
        )
    else:
        return jsonify({"error": "Invalid mode specified in JSON payload"}), 400

    response_payload = {
        "final_answer": final_reply,
        "usage_records": usage_data
    }
    return jsonify(response_payload), 200

# @app.route("/update_param", methods=["POST"])
# def update_param():
#     name  = request.form["name"]
#     dtype = request.form["dtype"]
#     shape = json.loads(request.form["shape"])
#     buf   = request.files["blob"].read()

#     # Send straight to the vLLM worker-extension
#     url = f"http://localhost:{service_args.small_model_port}/worker_extension/update_weight"
#     files = {"blob": buf}
#     data  = dict(name=name, dtype=dtype, shape=json.dumps(shape))
#     r = requests.post(url, data=data, files=files, timeout=600)
#     return jsonify(r.json()), r.status_code


# @app.route("/reset_cache", methods=["POST"])
# def reset_cache():
#     url = f"http://localhost:{service_args.small_model_port}/worker_extension/reset_cache"
#     r = requests.post(url, timeout=600)
#     return jsonify(r.json()), r.status_code

# ────────────────────────────────────────────────────────────
# helper: raw-bytes  →  torch.Tensor (on *this* GPU)
# ────────────────────────────────────────────────────────────
def _bytes_to_tensor(raw: bytes, dtype_str: str, shape: list[int]) -> torch.Tensor:
    arr = np.frombuffer(raw, dtype=np.dtype(dtype_str)).reshape(shape)
    return torch.from_numpy(arr).cuda()

# ────────────────────────────────────────────────────────────
#  POST /update_param
# ────────────────────────────────────────────────────────────

@app.route("/update_param", methods=["POST"])
def update_param():
    """
    ### WARNING
    This is VERY specific to our base 1.5B model, exercise caution!
    """
    if SMALL_LLM is None:
        return jsonify(error="small model not initialised"), 503

    name   = request.form["name"]
    dtype  = request.form["dtype"]                # "float32"|...|"uint16_bfloat"
    shape  = json.loads(request.form["shape"])
    blob   = request.files["blob"].read()

    def _swap(worker, p_name, dtype_tag, shp, raw):
        import torch, numpy as np

        model = (worker.model
                 if hasattr(worker, "model")
                 else worker.model_runner.model)

        # ---------------- reconstruct tensor --------------------------
        if dtype_tag == "uint16_bfloat":
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shp)
            t   = torch.from_numpy(arr).view(torch.bfloat16)
        else:
            arr = np.frombuffer(raw, dtype=np.dtype(dtype_tag)).reshape(shp)
            t   = torch.from_numpy(arr).to(dtype_tag)

        t = t.to(next(model.parameters()).device)
        # print name and shape etc of tensor
        print(f"[Worker {worker.rank}] Updating parameter '{p_name}' "
              f"to shape {t.shape} and dtype {t.dtype} on GPU {t.device}")


        # ───── map split-name → fused-name & slice ───────────────────────
        # recognise ...self_attn.{q,k,v}_proj.(weight|bias)
        import re
        m = re.match(r"^(.*self_attn\.)([qkv])_proj\.(weight|bias)$", p_name)
        if m:
            prefix, qkv, kind = m.groups()           # e.g. ("model.layers.0.self_attn.", "q", "weight")
            fused_name = f"{prefix}qkv_proj.{kind}"  # vLLM parameter
            fused_param = dict(model.named_parameters())[fused_name]

            hidden_size   = model.config.hidden_size        # 1536
            head_dim      = hidden_size // model.config.num_attention_heads  # 128
            kv_heads      = model.config.num_key_value_heads # 2
            kv_out        = kv_heads * head_dim              # 256
            q_slice, k_slice, v_slice = \
                slice(0, hidden_size), \
                slice(hidden_size, hidden_size + kv_out), \
                slice(hidden_size + kv_out, hidden_size + 2*kv_out)
            which = {"q": q_slice, "k": k_slice, "v": v_slice}[qkv]

            # copy into slice (dim-0 for weights / same for bias)
            with torch.no_grad():
                fused_param.data[which, ...].copy_(t)
            return 0   # done, skip the usual path below

        # ---------- gate / up mapping ----------
        m = re.match(r"^(.*mlp\.)(gate|up)_proj\.(weight|bias)$", p_name)
        if m:
            prefix, which, kind = m.groups()                 # "gate" or "up"
            fused_name = f"{prefix}gate_up_proj.{kind}"      # vLLM name
            fused_param = dict(model.named_parameters())[fused_name]

            inter_size = model.config.intermediate_size      # 8960
            sl = slice(0, inter_size) if which == "gate" else slice(inter_size, 2*inter_size)

            with torch.no_grad():
                fused_param.data[sl, ...].copy_(t)           # works for bias too
            return 0

        # ---------- 3) embed_tokens.weight  (handle padding) -------------
        if p_name.endswith("embed_tokens.weight"):
            target = dict(model.named_parameters())[p_name]
            rows   = min(target.size(0), t.size(0))   # 151 665
            with torch.no_grad():
                target.data[:rows].copy_(t[:rows])    # skip padded tail
            return 0

        # ---------- 4) ordinary 1-to-1 copy (shape-aware) ----------------
        target = dict(model.named_parameters())[p_name]
        if target.shape == t.shape:
            with torch.no_grad():
                target.data.copy_(t)
        elif target.ndim == t.ndim and target.shape[1:] == t.shape[1:]:
            # First-dim mismatch (e.g. embedding bias) – copy the overlap
            rows = min(target.size(0), t.size(0))
            with torch.no_grad():
                target.data[:rows].copy_(t[:rows])
            print(f"   ↳ shape mismatch; copied first {rows} rows")
        else:
            raise ValueError(
                f"Shape mismatch for {p_name}: target {target.shape}, incoming {t.shape}"
            )
        return 0

    SMALL_LLM.collective_rpc(_swap, args=(name, dtype, shape, blob))
    SMALL_LLM.reset_prefix_cache()
    return jsonify(status="ok"), 200


# ────────────────────────────────────────────────────────────
#  POST /reset_cache
# ────────────────────────────────────────────────────────────
@app.route("/reset_cache", methods=["POST"])
def reset_cache():
    if SMALL_LLM is None:
        return jsonify(error="small model not initialised"), 503

    SMALL_LLM.reset_prefix_cache()
    return jsonify(status="ok"), 200

    

##############################################################################
# Main entrypoint: parse arguments, possibly start big/small model servers
#                  then run the Flask app
##############################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--big_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--big_model_port", type=int, default=8000)
    parser.add_argument("--big_model_gpus", type=str, default="0,1")

    parser.add_argument("--small_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--small_model_port", type=int, default=8001)
    parser.add_argument("--small_model_gpus", type=str, default="2")
    ### Sequential scaling of big/small/logprob modes
    parser.add_argument("--sequential_scale", type=int, default=0)
    ### Modes, only 1 can be true ###
    parser.add_argument("--spec_reason", action="store_true")
    parser.add_argument("--spec_reason_perf", action="store_true")
    parser.add_argument("--spec_reason_perf_only", action="store_true")
    parser.add_argument("--placeholder_mode", action="store_true")
    parser.add_argument("--spec_rewrite", action="store_true")
    parser.add_argument("--random_switch", action="store_true")
    parser.add_argument("--logprob_subselect", action="store_true")
    parser.add_argument("--big_model_only", action="store_true")
    parser.add_argument("--small_model_only", action="store_true")
    ### End Of Modes, only 1 can be true ###
    parser.add_argument("--test_logging", action="store_true")

    parser.add_argument("--spec_decoding", action="store_true",help="Launch vLLM in OpenAI API mode with speculative_decoding")
    parser.add_argument("--speculative_config", type=str, default=None,help="JSON string for vLLM speculative_config (e.g. '{\"model\":...,\"num_speculative_tokens\":5,...}')")
    parser.add_argument("--seed", type=int, default=42,help="Random seed for speculative decoding server")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,help="vLLM GPU memory utilization fraction")
    parser.add_argument("--max_model_len", type=int, default=32768,help="Max model length for speculative decoding server")
    parser.add_argument("--enforce_eager", action="store_true",help="Pass --enforce-eager to the vLLM server")

    ### LogProb Subselect Args ###
    parser.add_argument("--sgen", type=int, default=8)
    parser.add_argument("--stok", type=int, default=16)
    parser.add_argument("--sdecay", type=int, default=2)
    parser.add_argument("--ltok", type=int, default=1)
    parser.add_argument("--lbound", type=int, default=4)
    parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of iterations, closesly controls max token budget.")
    ### End Of LogProb Subselect Args ###
    ### Random Switch Args ###
    parser.add_argument("--switch_ratio", type=int, default=1, help="Switching ratio, always 1:{switch_ratio}")
    parser.add_argument("--switch_chunk", type=int, default=16)
    ### End Of Random Switch Args ###
    ### Spec Rewrite Args ### FOR NOW, WE DONT HAVE ANY REQUIRED PARAMS ON SPEC REASONING, SO WE JUST PASS EVERYTHING FOR QUICK TESTING
    parser.add_argument("--full_rewrite", action="store_true")
    parser.add_argument("--draft_propose_ignore_str", action="store_true")
    parser.add_argument("--bloat_tokens", type=int, default=0)
    parser.add_argument("--thinking_n_ignore", type=int, default=2)
    parser.add_argument("--drafting_n", type=int, default=1)
    parser.add_argument("--small_first", action="store_true")
    ### End Of Spec Rewrite Args ###
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--terminating_string", type=str, default="\n Put your final answer within \\boxed{}.")
    parser.add_argument("--terminate_on_exit", action="store_true",
                        help="If True, will shut down vLLM servers on ctrl-c or exit.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port for this speculative reasoner Flask service.")
    return parser.parse_args()

def init_small_model_inproc(model_name: str, gpu_ids: str, max_len: int):
    """
    Launch the small vLLM model *inside* this process so that we can
    hot‑swap weights via Python, no HTTP.
    """
    global SMALL_LLM, SMALL_MODEL_RUNNER

    if SMALL_LLM is not None:   # already initialised
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    tp = len(gpu_ids.split(","))

    SMALL_LLM = LLM(
        model                 = model_name,
        tensor_parallel_size  = tp,
        trust_remote_code     = True,
        max_model_len         = max_len,
        enable_prefix_caching = True,
    )
    # from types import MethodType
    # # one dummy generate to force worker construction
    # _ = SMALL_LLM.generate(["ping"], SamplingParams(max_tokens=1, temperature=0.0))
    # def _apply_model_v1(self, func):
    #     executor = getattr(self.llm_engine, "model_executor",
    #                     getattr(self.llm_engine, "_executor", None))
    #     if executor is None:
    #         raise RuntimeError("No executor found on llm_engine")
    #     return executor.apply_model(func)
    # LLM.apply_model = MethodType(_apply_model_v1, LLM)    # patch class method


    # # -------- locate the model‑runner object -----------------
    # # V0 path:      SMALL_LLM._engine.model_runner
    # # V1 (0.8+) :   SMALL_LLM.engine._executor._workers[0].model_runner
    # runner = None
    # if hasattr(SMALL_LLM, "_engine"):           # vLLM ≤ 0.7
    #     runner = getattr(SMALL_LLM._engine, "model_runner", None)
    # if runner is None and hasattr(SMALL_LLM, "engine"):  # vLLM ≥ 0.8
    #     try:
    #         worker0 = SMALL_LLM.engine._executor._workers[0]
    #         runner  = worker0.model_runner
    #     except Exception:
    #         pass

    # if runner is None or not hasattr(runner, "model"):
    #     import pdb; pdb.set_trace()
    #     raise RuntimeError("Could not locate ModelRunner – weight‑sync unsupported.")

    # SMALL_MODEL_RUNNER = runner


def main():
    global big_model_proc, small_model_proc, service_args

    service_args = parse_args()

    # Determine which models to launch based on mode
    # — Pure speculative‐decoding mode —
    if service_args.spec_decoding:
        if service_args.speculative_config is None:
            print("[Service] ERROR: --speculative_config must be provided for speculative-decoding mode")
            sys.exit(1)

        print("[Service] Starting speculative-decoding mode …")
        big_model_proc = launch_spec_decoding_server(
            service_args.big_model,
            service_args.big_model_port,
            service_args.big_model_gpus,
            service_args.seed,
            service_args.gpu_memory_utilization,
            service_args.speculative_config,
            service_args.max_model_len,
            service_args.enforce_eager,
        )

        print("[Service] Waiting for speculative decoding server to be ready …")
        if not wait_for_server(f"http://localhost:{service_args.big_model_port}/ping"):
            print("[Service] Speculative decoding server did not come up in time. Exiting.")
            big_model_proc.terminate()
            sys.exit(1)
        print("[Service] Speculative decoding server is up.")

        # In pure‐decoding mode, we skip launching any other models
        need_big_model = False
        need_small_model = False

    else:
        if service_args.max_iterations is None:
            service_args.max_iterations = 32768 // (service_args.stok * service_args.ltok) 

        need_big_model = not service_args.small_model_only
        need_small_model = not service_args.big_model_only
    
    # Load tokenizer
    load_big_model_tokenizer(service_args.big_model)

    # 2) Check if small model is needed and launch if necessary

    if need_small_model:
        init_small_model_inproc(
            model_name  = service_args.small_model,
            gpu_ids     = service_args.small_model_gpus,
            max_len     = 16384,
        )
        print("[Service] Small model initialised *inside* Flask process ✅")
    # if need_small_model:
    #     small_model_proc = launch_small_model(service_args.small_model,
    #                                         service_args.small_model_port,
    #                                         service_args.small_model_gpus)
    #     print("[Service] Small model initialised in-process ✅")
    else:
        print("[Service] Small model is not needed. Skipping...")

    # 1) Check if the big model is needed and launch if necessary
    if need_big_model:
        big_model_proc = launch_big_model_vllm(service_args.big_model,
                                            service_args.big_model_port,
                                            service_args.big_model_gpus)
        print(f"[Service] Started big model; server is up on port {service_args.big_model_port} ...")
    else:
        print("[Service] Big model is not needed. Skipping...")

    
    ready_event.set()
    # 3) Start our Flask app
    try:
        app.run(host="0.0.0.0", port=service_args.port)
    except KeyboardInterrupt:
        pass
    finally:
        if service_args.terminate_on_exit:
            print("[Service] Terminating vLLM model servers because --terminate_on_exit was set.")
            if small_model_proc is not None and need_small_model:
                small_model_proc.send_signal(signal.SIGTERM)
                small_model_proc.wait()
            if big_model_proc is not None and need_big_model:
                big_model_proc.send_signal(signal.SIGTERM)
                big_model_proc.wait()
        print("[Service] Service shutting down.")


if __name__ == "__main__":
    main()