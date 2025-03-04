import argparse
import os
import subprocess
import time
import signal
import sys
import requests

from flask import Flask, request, jsonify

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
        "--enable_prefix_caching"
    ]
    print(f"[Service] Launching big model server on port {port} using GPUs {gpu_ids}")
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
        "--enable_prefix_caching"
    ]
    print(f"[Service] Launching small model (vLLM) server on port {port} using GPUs {gpu_ids}")
    return subprocess.Popen(cmd, env=env)

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


@app.route("/ping", methods=["GET"])
def ping():
    """
    Simple health check endpoint to verify service is up.
    """
    return jsonify({"message": "pong"}), 200


@app.route("/speculative_reason", methods=["POST"])
def speculative_reason():
    """
    This endpoint will accept a JSON payload that includes a 'prompt'
    or 'question', and then do multi-step CoT reasoning using big/small
    vLLM models launched in the background.

    Sample JSON POST to /speculative_reason:
        {
          "question": "What is 2+2?",
          "thinking_n_ignore": 2,
          "drafting_n": 1,
          "full_rewrite": false,
          "draft_propose_ignore_str": false,
          "max_tokens": 1024,
          "temperature": 0.7,
          "terminating_string": "\\nFinal Answer"
        }
    """
    data = request.json
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in JSON payload"}), 400

    question = data["question"]

    # We'll fallback to the service_args defaults if user didn't pass them.
    thinking_n_ignore = data.get("thinking_n_ignore", service_args.thinking_n_ignore)
    drafting_n = data.get("drafting_n", service_args.drafting_n)
    full_rewrite = data.get("full_rewrite", service_args.full_rewrite)
    temperature = data.get("temperature", 0.6)
    max_tokens = data.get("max_tokens", service_args.max_tokens)
    terminating_string = data.get("terminating_string", service_args.terminating_string)

    # If you want to do "bloat tokens," you could incorporate that logic here:
    if service_args.bloat_tokens > 0:
        bloat_sentence = ("question to follow soon. " * 20)  # ~100 tokens
        question = bloat_sentence * (service_args.bloat_tokens // 100) + question

    # Basic prompt for user query
    base_prompt = f"<|user|>\n{question}\n<|assistant|>\n"

    # Big model "thinking" format
    if "simplescaling" in service_args.big_model:
        big_model_think_prefix = "<|im_start|>think\n"
        big_model_think_suffix = "<|im_start|>answer"
    elif "deepseek-ai" in service_args.big_model:
        big_model_think_prefix = "<think>\n"
        big_model_think_suffix = "\n</think>"
    else:
        return jsonify({"error": "Unknown big model format."}), 400

    # Small model "thinking" format
    if "simplescaling" in service_args.small_model:
        small_model_think_prefix = "<|im_start|>think\n"
        small_model_think_suffix = "<|im_start|>answer"
    elif "deepseek-ai" in service_args.small_model:
        small_model_think_prefix = "<think>\n"
        small_model_think_suffix = "\n</think>"
    else:
        # For "drafting" models that do not have specialized tokens:
        small_model_think_prefix = "\n"
        small_model_think_suffix = "\n"

    wait_str = "\nWait"
    fallback_suffixes = ()

    cot_accumulator = ""
    usage_data = []

    # ---------------------------------------------------------------------
    # Step 1: multiple "thinking" iterations using either small or big model
    # ---------------------------------------------------------------------
    for i in range(thinking_n_ignore):
        # Decide whether to use big or small model this iteration
        if i == 0 and service_args.small_first:
            model_port = service_args.small_model_port
            model_name = service_args.small_model
            model_think_prefix = small_model_think_prefix
            model_think_suffix = small_model_think_suffix
        else:
            model_port = service_args.big_model_port
            model_name = service_args.big_model
            model_think_prefix = big_model_think_prefix
            model_think_suffix = big_model_think_suffix

        # Prompt includes previously accumulated CoT
        if i == 0:
            iteration_prompt = base_prompt + model_think_prefix
        else:
            iteration_prompt = base_prompt + cot_accumulator

        raw_resp, latency = generate_text_vllm(
            iteration_prompt,
            port=model_port,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model_name
        )
        # usage info (if the server returns it)
        usage_dict = raw_resp.get("usage", {})
        usage_data.append({
            "Model": model_name,
            "ThinkIter": i+1,
            "DraftVersion": 0,
            "PromptTokens": usage_dict.get("prompt_tokens", 0),
            "CompletionTokens": usage_dict.get("completion_tokens", 0),
            "Latency": latency
        })

        raw_reply = raw_resp["choices"][0]["text"]
        partial_cot = extract_cot(raw_reply, model_think_suffix, fallback_suffixes)

        # Optionally do drafting with the small model
        if drafting_n > 0:
            drafts = []
            for d_i in range(drafting_n):
                if full_rewrite:
                    prompt_for_draft = (
                        f"The question asked:\n{question}\n\n"
                        f"We currently have partial reasoning:\n\n"
                        f"{cot_accumulator} \t {partial_cot}\n"
                        f"{small_model_think_prefix}"
                        "I want to concisely refine the above reasoning, preserving ALL crucial steps."
                    )
                else:
                    if len(cot_accumulator) == 0:
                        cot_inject = "No Partial CoT yet"
                    else:
                        cot_inject = cot_accumulator
                    prompt_for_draft = (
                        f"The question asked:\n{question}\n\n"
                        f"Prior reasoning chain:\n{cot_inject}\n\n"
                        f"Recently, we have partial reasoning:\n\n"
                        f"{partial_cot}\n"
                        f"{small_model_think_prefix}"
                        "I want to refine ONLY this partial reasoning. "
                    )
                if service_args.draft_propose_ignore_str:
                    prompt_for_draft += (
                        "I must also conclude with a question on the reasoning that encourages deeper investigation "
                        "to make the steps of solution more robust."
                        f"{small_model_think_suffix}"
                    )
                else:
                    prompt_for_draft += f"{small_model_think_suffix}"

                draft_resp, draft_latency = generate_text_vllm(
                    prompt_for_draft,
                    port=service_args.small_model_port,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=service_args.small_model
                )
                usage_dict = draft_resp.get("usage", {})
                usage_data.append({
                    "Model": service_args.small_model,
                    "ThinkIter": i+1,
                    "DraftVersion": d_i+1,
                    "PromptTokens": usage_dict.get("prompt_tokens", 0),
                    "CompletionTokens": usage_dict.get("completion_tokens", 0),
                    "Latency": draft_latency
                })

                raw_draft = draft_resp["choices"][0]["text"]
                drafts.append(raw_draft)

            # pick "best" draft by some scoring
            best_score = -1e9
            best_draft = drafts[0]
            for d in drafts:
                # e.g. trivial score: length of text
                s = len(d)
                if s > best_score:
                    best_score = s
                    best_draft = d
            partial_cot = best_draft

        # Accumulate partial chain-of-thought
        if full_rewrite:
            cot_accumulator = partial_cot + wait_str
        else: # No longer checks for draft-propose-ignore-str, better to just add wait anyway.
            cot_accumulator += partial_cot + wait_str

    # ---------------------------------------------------------------------
    # Step 2: final answer
    # ---------------------------------------------------------------------
    final_prompt = base_prompt + terminating_string + cot_accumulator
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
        "ThinkIter": "final",
        "DraftVersion": 0,
        "PromptTokens": usage_dict.get("prompt_tokens", 0),
        "CompletionTokens": usage_dict.get("completion_tokens", 0),
        "Latency": final_latency
    })

    final_reply = final_resp["choices"][0]["text"]
    response_payload = {
        "final_answer": final_reply,
        "usage_records": usage_data
    }
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

    parser.add_argument("--small_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--small_model_port", type=int, default=8001)
    parser.add_argument("--small_model_gpus", type=str, default="2")

    parser.add_argument("--thinking_n_ignore", type=int, default=2)
    parser.add_argument("--drafting_n", type=int, default=1)
    parser.add_argument("--small_first", action="store_true")
    parser.add_argument("--full_rewrite", action="store_true")
    parser.add_argument("--draft_propose_ignore_str", action="store_true")
    parser.add_argument("--bloat_tokens", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--terminating_string", type=str, default="\nPut your final answer within \\boxed{}.")
    parser.add_argument("--terminate_on_exit", action="store_true",
                        help="If True, will shut down vLLM servers on ctrl-c or exit.")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port for this speculative reasoner Flask service.")
    return parser.parse_args()


def main():
    global big_model_proc, small_model_proc, service_args

    service_args = parse_args()

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