import argparse
import os
import subprocess
import time
import signal
import sys
import requests
import threading

from flask import Flask, request, jsonify
from typing import List, Tuple, Dict, Any, Optional, Union

from modes.spec_rewrite import run_speculative_reason_flow
from modes.placeholder import run_placeholder_flow
from modes.logprob_subselect import run_logprob_subselect_flow
from modes.small_model_only import run_smallmodel_flow
from modes.big_model_only import run_bigmodel_flow
from modes.random_switch_flow import run_random_switch_flow

app = Flask(__name__)

big_model_proc = None
small_model_proc = None
service_args = None   # Will hold the parsed arguments

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

def launch_big_model_vllm(big_model, port, gpu_ids):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    tp_size = len(gpu_ids.split(","))

    cmd = [
        "vllm", "serve",
        big_model,
        "--port", str(port),
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", "32768",
        "--uvicorn-log-level=warning",
        # "--max-num-batched-tokens", "32768",
        "--enable_prefix_caching",
        "--enable-chunked-prefill"
    ]
    print(f"[Service] Launching big model server on port {port} using GPUs {gpu_ids} with **PrefixCaching AND ChunkedPrefill**")
    return subprocess.Popen(cmd, env=env)

def launch_small_model(model_name, port, gpu_ids):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    tp_size = len(gpu_ids.split(","))

    cmd = [
        "vllm", "serve",
        model_name,
        "--port", str(port),
        "--trust-remote-code",
        "--tensor-parallel-size", str(tp_size),
        "--max-model-len", "32768",
        "--uvicorn-log-level=warning",
        # "--max-num-batched-tokens", "32768",
        "--enable_prefix_caching",
        "--enable-chunked-prefill"
    ]
    print(f"[Service] Launching small model (vLLM) server on port {port} using GPUs {gpu_ids} with **PrefixCaching AND ChunkedPrefill**")
    return subprocess.Popen(cmd, env=env)

def batched_generate_text_vllm(
    prompts: List[str], 
    port: int = 8000, 
    temperature: float = 0.6, 
    max_tokens: int = 128, 
    model: str = "my-model", 
    requests=None
) -> Tuple[List[dict], float]:
    """
    A batched call to the vLLM HTTP server's /v1/completions endpoint.
    - prompts: List of prompt strings (one per item).
    - port, temperature, max_tokens, model: same as before.
    - requests: the requests library (or a compatible mock).

    Returns:
      - A list of response JSON objects (each corresponding to an item in `prompts`).
      - A single float indicating the average latency for the entire batch.
    """
    if requests is None:
        import requests as _requests
        requests = _requests

    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompts,         # pass the entire list of prompts at once
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": 1,                    # generate 1 completion per prompt
    }

    start_time = time.time()
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    end_time = time.time()

    resp_json = resp.json()

    # The 'choices' key in vLLM is typically a flat list of size=len(prompts), each with a "text".
    # We have to parse them carefully to associate each choice with the correct prompt index.
    # By default, the OpenAI-compatible /v1/completions returns one choice per prompt in order.
    # So resp_json["choices"][i] corresponds to prompts[i].
    choices = resp_json["choices"]

    # Build a list of response dicts, each containing exactly what a normal single-call would have returned.
    # For consistency with your single-call usage, we mimic the structure:
    # Each item will have the structure of {"choices": [ { "text": ..., ... } ]}
    results = []
    for i, choice in enumerate(choices):
        results.append({
            "choices": [choice]
        })

    total_latency = end_time - start_time
    avg_latency = total_latency / max(1, len(prompts))

    return results, avg_latency

def batched_eval_logprob_vllm(
    text_batch: List[str],
    big_model_port: int,
    big_model: str,
    requests,
    temperature: float = 0.0,
    max_tokens: int = 0,
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
    
def generate_text_vllm(prompt, port=8000, temperature=0.6, max_tokens=128, model="my-model"):
    """
    A direct call to the vLLM HTTP server's /v1/completions endpoint.
    """
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    start_time = time.time()
    resp = requests.post(url, json=payload)
    end_time = time.time()
    resp.raise_for_status()
    latency = end_time - start_time
    return resp.json(), latency

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

@app.route("/ping", methods=["GET"])
def ping():
    """
    Simple health check endpoint to verify service is up.
    """
    return jsonify({"message": "pong"}), 200

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

    if data.get("placeholder_mode", False):
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
            sequential_scale=data.get("sequential_scale", service_args.sequential_scale)
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
            sequential_scale=data.get("sequential_scale", service_args.sequential_scale)
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
        final_reply, usage_data = run_speculative_reason_flow(
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
    else:
        return jsonify({"error": "Invalid mode specified in JSON payload"}), 400

    response_payload = {
        "final_answer": final_reply,
        "usage_records": usage_data
    }
    return jsonify(response_payload), 200

@app.route("/think_summarize", methods=["POST"])
def think_summarize():
    """
    This endpoint implements a different reasoning approach where the model:
    1. Thinks for a while (generates some tokens)
    2. Summarizes its thoughts
    3. Uses these summaries for the next round of thinking
    4. Repeats this process multiple times
    5. Generates a final answer based on the accumulated summaries

    Sample JSON POST to /think_summarize:
        {
          "question": "What is 2+2?",
          "think_tokens": 200,
          "num_rounds": 3,
          "max_tokens": 1024,
          "temperature": 0.7,
          "debug_prompts": true,
          "terminating_string": "\\nFinal Answer"
        }
    """
    data = request.json
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in JSON payload"}), 400

    question = data["question"]
    test_logging = data.get("test_logging", False)
    if test_logging:
        draft_logs = "draft_logs"
        if not os.path.exists(draft_logs):
            os.makedirs(draft_logs)
        print(f"[Service] test_logging=True; extra debug for question: {question}")

    # Get parameters with defaults
    think_tokens = data.get("think_tokens", 200)
    num_rounds = data.get("num_rounds", 3)
    temperature = data.get("temperature", 0.6)
    max_tokens = data.get("max_tokens", service_args.max_tokens)
    terminating_string = data.get("terminating_string", service_args.terminating_string)
    debug_prompts = data.get("debug_prompts", False)

    # Basic prompt for user query
    base_prompt = f"<|user|>\n{question}\n<|assistant|>\n"

    # Big model format for thinking
    if "simplescaling" in service_args.big_model:
        big_model_think_prefix = "<|im_start|>think\n"
        big_model_think_suffix = "<|im_start|>answer"
    elif "deepseek-ai" in service_args.big_model:
        big_model_think_prefix = "<think>\n"
        big_model_think_suffix = "\n</think>"
    else:
        return jsonify({"error": "Unknown big model format."}), 400

    cot_accumulator = ""
    usage_data = []
    
    if debug_prompts:
        print("\n=== Starting Think-Summarize Process ===")
        print(f"Number of rounds: {num_rounds}")
        print(f"Tokens per thinking phase: {think_tokens}")
        print("\nInitial prompt:")
        print(base_prompt)

    for round_idx in range(num_rounds):
        if debug_prompts:
            print(f"\n=== Round {round_idx + 1} ===")

        # Thinking phase
        think_prompt = base_prompt + cot_accumulator + big_model_think_prefix
        if debug_prompts:
            print("\nThinking prompt:")
            print(think_prompt)

        think_resp, think_latency = generate_text_vllm(
            think_prompt,
            port=service_args.big_model_port,
            temperature=temperature,
            max_tokens=think_tokens,
            model=service_args.big_model
        )
        usage_dict = think_resp.get("usage", {})
        usage_data.append({
            "Model": service_args.big_model,
            "Round": round_idx + 1,
            "Phase": "think",
            "PromptTokens": usage_dict.get("prompt_tokens", 0),
            "CompletionTokens": usage_dict.get("completion_tokens", 0),
            "Latency": think_latency
        })

        thoughts = think_resp["choices"][0]["text"]
        if debug_prompts:
            print("\nThoughts generated:")
            print(thoughts)

        # Summarization phase
        summary_prompt = base_prompt + cot_accumulator + thoughts + "\n\nIn summary:"
        if debug_prompts:
            print("\nSummary prompt:")
            print(summary_prompt)

        summary_resp, summary_latency = generate_text_vllm(
            summary_prompt,
            port=service_args.big_model_port,
            temperature=temperature,
            max_tokens=max_tokens // 4,  # Use fewer tokens for summary
            model=service_args.big_model
        )
        usage_dict = summary_resp.get("usage", {})
        usage_data.append({
            "Model": service_args.big_model,
            "Round": round_idx + 1,
            "Phase": "summarize",
            "PromptTokens": usage_dict.get("prompt_tokens", 0),
            "CompletionTokens": usage_dict.get("completion_tokens", 0),
            "Latency": summary_latency
        })

        summary = summary_resp["choices"][0]["text"]
        if debug_prompts:
            print("\nSummary generated:")
            print(summary)

        # Update accumulator with just the summary
        cot_accumulator += f"Round {round_idx + 1} summary: {summary}\n\n"

    # Final answer phase
    final_prompt = base_prompt + terminating_string + cot_accumulator
    if debug_prompts:
        print("\n=== Final Answer Phase ===")
        print("Final prompt:")
        print(final_prompt)

    final_resp, final_latency = generate_text_vllm(
        final_prompt,
        port=service_args.big_model_port,
        temperature=temperature,
        max_tokens=max_tokens,
        model=service_args.big_model
    )
    usage_dict = final_resp.get("usage", {})
    usage_data.append({
        "Model": service_args.big_model,
        "Round": "final",
        "Phase": "answer",
        "PromptTokens": usage_dict.get("prompt_tokens", 0),
        "CompletionTokens": usage_dict.get("completion_tokens", 0),
        "Latency": final_latency
    })

    final_reply = final_resp["choices"][0]["text"]
    if debug_prompts:
        print("\nFinal answer generated:")
        print(final_reply)

    response_payload = {
        "final_answer": final_reply,
        "usage_records": usage_data,
        "summaries": cot_accumulator  # Include the accumulated summaries in response
    }

    if test_logging:
        write_model_name = service_args.big_model.split("/")[-1]
        with open(f"{draft_logs}/think_summarize_{write_model_name}.txt", "w") as f:
            f.write("\n" + "-" * 80 + "\n" + "\n" + "Final Prompt" + "\n" + "\n" + "-" * 80 + "\n")
            f.write(final_prompt)
            f.write("\n" + "-" * 80 + "\n"+ "\n" + "Final Continuation" + "\n" + "\n" + "-" * 80 + "\n")
            f.write(final_reply)
            f.write("\n" + "-" * 80 + "\n"+ "\n" + "Accumulated Summaries" + "\n" + "\n" + "-" * 80 + "\n")
            f.write(cot_accumulator)
        print("[Service] Usage data:", usage_data)

    return jsonify(response_payload), 200


##############################################################################
# Main entrypoint: parse arguments, possibly start big/small model servers
#                  then run the Flask app
##############################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--big_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    parser.add_argument("--big_model_port", type=int, default=8000)
    parser.add_argument("--big_model_gpus", type=str, default="0,1")

    parser.add_argument("--small_model", type=str, default="")
    parser.add_argument("--small_model_port", type=int, default=8001)
    parser.add_argument("--small_model_gpus", type=str, default="2")
    ### Sequential scaling of big/small/logprob modes
    parser.add_argument("--sequential_scale", type=int, default=0)
    ### Modes, only 1 can be true ###
    parser.add_argument("--placeholder_mode", action="store_true")
    parser.add_argument("--spec_rewrite", action="store_true")
    parser.add_argument("--random_switch", action="store_true")
    parser.add_argument("--logprob_subselect", action="store_true")
    parser.add_argument("--big_model_only", action="store_true")
    parser.add_argument("--small_model_only", action="store_true")
    ### End Of Modes, only 1 can be true ###
    parser.add_argument("--test_logging", action="store_true")
    ### LogProb Subselect Args ###
    parser.add_argument("--sgen", type=int, default=8)
    parser.add_argument("--stok", type=int, default=16)
    parser.add_argument("--sdecay", type=int, default=2)
    parser.add_argument("--ltok", type=int, default=0)
    parser.add_argument("--lbound", type=int, default=4)
    parser.add_argument("--max_iterations", type=int, default=None, help="Maximum number of iterations, closesly controls max token budget.")
    ### End Of LogProb Subselect Args ###
    ### Random Switch Args ###
    parser.add_argument("--switch_ratio", type=int, default=1, help="Switching ratio, always 1:{switch_ratio}")
    parser.add_argument("--switch_chunk", type=int, default=16)
    ### End Of Random Switch Args ###
    ### Spec Rewrite Args ###
    parser.add_argument("--full_rewrite", action="store_true")
    parser.add_argument("--draft_propose_ignore_str", action="store_true")
    parser.add_argument("--bloat_tokens", type=int, default=0)
    parser.add_argument("--thinking_n_ignore", type=int, default=2)
    parser.add_argument("--drafting_n", type=int, default=1)
    parser.add_argument("--small_first", action="store_true")
    ### End Of Spec Rewrite Args ###
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--terminating_string", type=str, default=" \n Put your final answer within \\boxed{}.")
    parser.add_argument("--terminate_on_exit", action="store_true",
                        help="If True, will shut down vLLM servers on ctrl-c or exit.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port for this speculative reasoner Flask service.")
    return parser.parse_args()


def main():
    global big_model_proc, small_model_proc, service_args

    service_args = parse_args()
    if service_args.max_iterations is None:
        service_args.max_iterations = 32768 // (service_args.stok * service_args.ltok) 

    # 1) Check if the big model is up. If not, we launch it
    print(f"[Service] Checking if big model server is up on port {service_args.big_model_port} ...")
    if wait_for_server(f"http://localhost:{service_args.big_model_port}/ping", timeout=5):
        print("[Service] Big model is already up.")
    else:
        big_model_proc = launch_big_model_vllm(service_args.big_model,
                                               service_args.big_model_port,
                                               service_args.big_model_gpus)
        print("[Service] Waiting for big model server to be ready ...")
        if not wait_for_server(f"http://localhost:{service_args.big_model_port}/ping"):
            print("[Service] Big model server did not come up in time. Exiting.")
            if big_model_proc is not None:
                big_model_proc.terminate()
            sys.exit(1)
        print("[Service] Big model server is up.")


    if service_args.small_model == "":
        print("[Service] No small model specified. Skipping small model launch.")
    else:
        # 2) Check if small model is up. If not, we launch it
        print(f"[Service] Checking if small model server is up on port {service_args.small_model_port} ...")
        if wait_for_server(f"http://localhost:{service_args.small_model_port}/ping", timeout=5):
            print("[Service] Small model is already up.")
        else:
            small_model_proc = launch_small_model(service_args.small_model,
                                                service_args.small_model_port,
                                                service_args.small_model_gpus)
            print("[Service] Waiting for small model server to be ready ...")
            if not wait_for_server(f"http://localhost:{service_args.small_model_port}/ping"):
                print("[Service] Small model server did not come up in time. Exiting.")
                if small_model_proc is not None:
                    small_model_proc.terminate()
                if big_model_proc is not None:
                    big_model_proc.terminate()
                sys.exit(1)
            print("[Service] Small model server is up.")

    # 3) Start our Flask app
    try:
        app.run(host="0.0.0.0", port=service_args.port)
    except KeyboardInterrupt:
        pass
    finally:
        if service_args.terminate_on_exit:
            print("[Service] Terminating vLLM model servers because --terminate_on_exit was set.")
            if small_model_proc is not None:
                small_model_proc.send_signal(signal.SIGTERM)
                small_model_proc.wait()
            if big_model_proc is not None:
                big_model_proc.send_signal(signal.SIGTERM)
                big_model_proc.wait()
        print("[Service] Service shutting down.")


if __name__ == "__main__":
    main()