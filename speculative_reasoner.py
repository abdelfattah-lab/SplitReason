import argparse
import subprocess
import time
import requests
import signal
import sys
import os
from tabulate import tabulate


################################################################################
# Parse arguments
################################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--big_model", type=str, default="simplescaling/s1-32B",
                        help="The large model to serve via vLLM.")
    parser.add_argument("--big_model_port", type=int, default=8000,
                        help="TCP port for the big model server.")
    # parser.add_argument("--big_model_gpus", type=str, default="0,1,2,3",
    parser.add_argument("--big_model_gpus", type=str, default="0,1",
                        help="Comma-separated GPU IDs for the big model (env-based).")

    parser.add_argument("--small_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                        help="The small draft model to serve (vLLM).")
    parser.add_argument("--small_model_port", type=int, default=8001,
                        help="TCP port for the draft model server.")
    # parser.add_argument("--small_model_gpus", type=str, default="4,5",
    parser.add_argument("--small_model_gpus", type=str, default="2",
                        help="Comma-separated GPU IDs for the small model (env-based).")

    parser.add_argument("--thinking_n_ignore", type=int, default=2,
                        help="Number of chain-of-thought (CoT) iterations to do before final.")
    parser.add_argument("--drafting_n", type=int, default=2,
                        help="Number of parallel drafts to generate from the small model.")
    parser.add_argument("--small_first", action="store_true",
                        help="If True, use the small model for the initial CoT, otherwise use big model.")
    parser.add_argument("--draft_propose_ignore_str", action="store_true",
                        help="If True, the draft model proposes an 'Ignore String' to the big model.")
    # parser.add_argument("--question", type=str, default="How many rs in strawberry?",
                        # help="Test question to send to big model and small model.")
    parser.add_argument("--question", type=str, default="Jen enters a lottery by picking $4$ distinct numbers from $S=\{1,2,3,\cdots,9,10\}.$ $4$ numbers are randomly chosen from $S.$ She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.", help="Question to answer")

    return parser.parse_args()

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
        "--max-model-len", "32768"
    ]
    print(f"Launching big model server on port {port} using GPUs {gpu_ids}")
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
        "--max-model-len", "32768"
    ]
    print(f"Launching small model (vLLM) server on port {port} using GPUs {gpu_ids}")
    return subprocess.Popen(cmd, env=env)

def generate_text_vllm(prompt, port=8000, temperature=0.6, max_tokens=128, model="my-model"):
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()

def extract_cot(raw_text, think_suffix, fallback_suffixes=()):
    """
    Attempt to split the model's reply to isolate the chain-of-thought portion.
    If we find 'think_suffix' in the text, we split on it and take what's before.
    Otherwise, if we find any fallback in fallback_suffixes, we do the same.
    If nothing is found, return the entire text.
    """
    if think_suffix and think_suffix in raw_text:
        return raw_text.split(think_suffix)[0]

    for fb in fallback_suffixes:
        if fb in raw_text:
            return raw_text.split(fb)[0]

    return raw_text

def score_with_big_model(draft_text, big_model, big_model_port):
    # For now, we simply return the length: longer might be "better"
    return len(draft_text)

def record_usage(usage_data, model_name, think_iter, draft_version, usage_dict):
    usage_data.append({
        "Model": model_name,
        "ThinkIter": think_iter,        
        "DraftVersion": draft_version,  
        "PromptTokens": usage_dict.get("prompt_tokens", 0),
        "CompletionTokens": usage_dict.get("completion_tokens", 0)
    })

def print_usage_table(usage_data):
    rows = []
    for entry in usage_data:
        rows.append([
            entry["Model"],
            entry["ThinkIter"],
            entry["DraftVersion"],
            entry["PromptTokens"],
            entry["CompletionTokens"]
        ])
    print("\n=== Token Usage Summary ===")
    print(tabulate(
        rows,
        headers=["Model", "ThinkIter", "DraftVersion", "PromptTokens", "CompletionTokens"],
        tablefmt="grid"
    ))

def main():
    draft_logs = "draft_logs"
    if not os.path.exists(draft_logs):
        os.makedirs(draft_logs)

    args = parse_args()

    usage_data = []
    big_model_proc = launch_big_model_vllm(args.big_model,
                                           args.big_model_port,
                                           args.big_model_gpus)
    print("Waiting for big model server to be ready...")
    if not wait_for_server(f"http://localhost:{args.big_model_port}/ping"):
        print("Big model server did not come up within time limit.")
        big_model_proc.terminate()
        sys.exit(1)
    print("Big model server is up.")

    small_model_proc = launch_small_model(args.small_model,
                                          args.small_model_port,
                                          args.small_model_gpus)
    print("Waiting for small model server to be ready...")
    if not wait_for_server(f"http://localhost:{args.small_model_port}/ping"):
        print("Small model server did not come up within time limit.")
        small_model_proc.terminate()
        big_model_proc.terminate()
        sys.exit(1)
    print("Small model server is up.")

    # Basic prompt for user query
    base_prompt = f"<|user|>\n{args.question}\n<|assistant|>\n"
    # Big model think tokens
    if "simplescaling" in args.big_model:
        big_model_think_prefix = "<|im_start|>think\n"
        big_model_think_suffix = "<|im_start|>answer"
    elif "deepseek-ai" in args.big_model:
        big_model_think_prefix = "<think>\n"
        big_model_think_suffix = "\n</think>"
    else:
        raise ValueError("Unknown big model type -- it must be a 'thinking' model.")

    # Small model think tokens
    if "simplescaling" in args.small_model:
        small_model_think_prefix = "<|im_start|>think\n"
        small_model_think_suffix = "<|im_start|>answer"
    elif "deepseek-ai" in args.small_model:
        small_model_think_prefix = "<think>\n"
        small_model_think_suffix = "\n</think>"
    else: # Drafting model might not be a 'thinking' model
        small_model_think_prefix = "\n"
        small_model_think_suffix = "\n"

    wait_str = "\nWait"

    fallback_suffixes = ("\nanswer", "\nAnswer", "**Final Answer**", "\nFinal Answer", "<|answer|>")
    # We'll accumulate the partial CoT across iterations
    # At each iteration, we either generate from big or small, parse the CoT,
    # then do drafting with small model, pick best draft, append "Wait".

    cot_accumulator = ""

    for i in range(args.thinking_n_ignore):
        print(f"\n=== Iteration {i+1}/{args.thinking_n_ignore} ===")

        # Decide whether to use small or big model for the initial generation
        if i == 0 and args.small_first:
            print("Using small model for the first CoT generation.")
            model_port = args.small_model_port
            model_name = args.small_model
            model_think_prefix = small_model_think_prefix
            model_think_suffix = small_model_think_suffix
        else:
            print("Using big model for CoT generation.")
            model_port = args.big_model_port
            model_name = args.big_model
            model_think_prefix = big_model_think_prefix
            model_think_suffix = big_model_think_suffix

        # Full prompt for this iteration includes the partial CoT so far
        # plus the new "think" prefix.
        if i == 0: # No CoT yet, so we start fresh and start the 'thinking' mode.
            iteration_prompt = base_prompt + model_think_prefix
        else: # No need to add think prefix if already in thinking mode.
            iteration_prompt = base_prompt + cot_accumulator

        print(f"Prompt to {model_name}:\n{iteration_prompt}\n---")

        # Generate partial CoT from chosen model
        raw_resp = generate_text_vllm(
            iteration_prompt,
            port=model_port,
            temperature=0.6,
            max_tokens=16384,
            model=model_name
        )
        if "usage" in raw_resp:
            record_usage(usage_data, model_name, think_iter=(i+1), draft_version=0, usage_dict=raw_resp["usage"])

        raw_reply = raw_resp["choices"][0]["text"]

        # Extract chain-of-thought portion from the raw reply
        partial_cot = extract_cot(raw_reply, model_think_suffix, fallback_suffixes)
        print(f"Partial CoT extracted:\n{partial_cot}\n---")

        write_model_name = model_name.split("/")[-1]
        with open(f"{draft_logs}/{write_model_name}_iter{i+1}.txt", "w") as f:
            f.write(iteration_prompt)
            f.write("\n" + "-" * 80 + "\n")
            f.write("\n" + "Raw Reply" + "\n")
            f.write(raw_reply)
            f.write("\n" + "-" * 80 + "\n")
            f.write("\n" + "Extracted CoT" + "\n")
            f.write("\n" + "-" * 80 + "\n")
            f.write(partial_cot)
        print(f"[{model_name} raw reply]:\n{raw_reply}\n---")

        drafts = []
        if args.drafting_n > 0:
            print(f"Generating {args.drafting_n} draft variants from small model.")
            for d_i in range(args.drafting_n):
                prompt_for_draft = (
                    f"The question asked:\n{args.question}\n"
                    f"For the solution, we currently have this partial reasoning:\n"
                    f"{partial_cot}\n"
                    f"{small_model_think_prefix}I want to rewrite and refine the above reasoning, preserving all crucial steps "
                )
                if args.draft_propose_ignore_str:
                    prompt_for_draft += (
                        "Conclude with a leading question that encourages deeper investigation "
                        "into how to finalize the solution or verify the steps."
                        f"{small_model_think_suffix}"
                    )
                else:
                    prompt_for_draft += f"{small_model_think_suffix}"
                raw_draft_resp = generate_text_vllm(
                    prompt_for_draft,
                    port=args.small_model_port,
                    temperature=0.6,
                    max_tokens=16384,
                    model=args.small_model
                )
                if "usage" in raw_draft_resp:
                    record_usage(
                        usage_data,
                        args.small_model,
                        think_iter=(i+1),       
                        draft_version=(d_i+1),  
                        usage_dict=raw_draft_resp["usage"]
                    )

                raw_draft = raw_draft_resp["choices"][0]["text"]
                write_model_name = args.small_model.split("/")[-1]
                with open(f"{draft_logs}/{write_model_name}_iter{i+1}_draft{d_i+1}.txt", "w") as f:
                    f.write(prompt_for_draft)
                    f.write("\n" + "-" * 80 + "\n")
                    f.write(raw_draft)
                drafts.append(raw_draft)

            best_score = -1e9
            best_draft = drafts[0]
            for d in drafts:
                s = score_with_big_model(d, args.big_model, args.big_model_port)
                if s > best_score:
                    best_score = s
                    best_draft = d
            print("Selected best draft:\n", best_draft, "\n---")
            partial_cot = best_draft
        
        if args.draft_propose_ignore_str:
            cot_accumulator += partial_cot
        else:
            cot_accumulator += partial_cot + wait_str

    final_prompt = (
        base_prompt + cot_accumulator
        # + "<|im_start|>answer\n"
    )
    print("\n=== Final prompt to big model ===\n", final_prompt, "\n---")

    final_resp = generate_text_vllm(
        final_prompt,
        port=args.big_model_port,
        temperature=0.6,
        max_tokens=1024,
        model=args.big_model
    )
    if "usage" in final_resp:
        record_usage(usage_data, args.big_model, think_iter="final", draft_version=0, usage_dict=final_resp["usage"])

    final_reply = final_resp["choices"][0]["text"]
    write_model_name = args.big_model.split("/")[-1]
    with open(f"{draft_logs}/final_reply_{write_model_name}.txt", "w") as f:
        f.write(final_prompt)
        f.write("\n" + "-" * 80 + "\n")
        f.write(final_reply)
    print("[Big Model Final Reply]:\n", final_reply)


    print_usage_table(usage_data)
    print("Terminating servers.")
    small_model_proc.send_signal(signal.SIGTERM)
    big_model_proc.send_signal(signal.SIGTERM)
    small_model_proc.wait()
    big_model_proc.wait()
    print("All done.")

if __name__ == "__main__":
    main()
