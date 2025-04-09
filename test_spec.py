#!/usr/bin/env python

import argparse
import os
import time
import subprocess
import sys
import requests
from tabulate import tabulate

def ping_service(host, port, timeout=3.0):
    """Check if /ping endpoint returns 200 OK."""
    url = f"http://{host}:{port}/ping"
    try:
        r = requests.get(url, timeout=timeout)
        return (r.status_code == 200)
    except Exception:
        return False

def wait_for_service(host, port, timeout=600):
    """Wait up to `timeout` seconds for the /ping to succeed."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if ping_service(host, port):
            return True
        time.sleep(2)
    return False

def print_usage_table(usage_data):
    rows = []
    for entry in usage_data:
        model = entry.get("Model", "")
        think_iter = entry.get("ThinkIter", "")
        draft_ver = entry.get("DraftVersion", "")
        prompt_toks = entry.get("PromptTokens", 0)
        completion_toks = entry.get("CompletionTokens", 0)
        latency = entry.get("Latency", 0.0)
        if completion_toks > 0:
            per_token_lat = latency / completion_toks
        else:
            per_token_lat = 0.0
        rows.append([
            model,
            think_iter,
            draft_ver,
            prompt_toks,
            completion_toks,
            round(latency, 2),
            round(per_token_lat, 4),
        ])
    print("\n=== Token Usage Summary ===")
    print(
        tabulate(
            rows,
            headers=[
                "Model",
                "ThinkIter",
                "DraftVersion",
                "PromptToks",
                "CompleteToks",
                "Latency(s)",
                "Lat(s)/Tok"
            ],
            tablefmt="grid",
        )
    )
    print()

def main():
    parser = argparse.ArgumentParser(
        description="Test runner for spec_service.py with a single question."
    )
    # parser.add_argument("--question", type=str, default="The wheel shown is spun twice, so that the numbers indicated by the pointer are randomly determined (with each number on the wheel being equally likely). The two numbers determined in this way are recorded. The first number is divided by 4, determining one of the remainders 1,2,3 marking the columns of the checkerboard shown. The second number is divided by 5, determining one of the remainders 1,2,3,4 marking the rows of the checkerboard. Finally, a checker is placed on the square where this column and row meet. What is the probability that the checker is placed on a shaded square of the checkerboard? \n \n[asy] \nunitsize(1cm); \ndraw(Circle((0,0),2),linewidth(0.7)); \ndraw((1.7,1)--(-1.7,-1),linewidth(0.7)); \ndraw((1.7,-1)--(-1.7,1),linewidth(0.7)); \ndraw((0,2)--(0,-2)); \nlabel("1",(0.8,0.5),NW); \nlabel("2",(0.8,-0.5),SW); \nlabel("6",(-0.8,0.5),NE); \nlabel("9",(-0.8,-0.5),SE); \nlabel("3",(-0.7,0),W); \nlabel("7",(0.7,0),E); \ndraw((-2.8,0)--(-2.1,0),Arrow); \nlabel("Pointer",(-2.8,0),W); \nfill((3,0)--(3,1)--(4,1)--(4,0)--cycle,gray(0.7)); \nfill((3,-2)--(3,-1)--(4,-1)--(4,-2)--cycle,gray(0.7)); \nfill((4,1)--(4,2)--(5,2)--(5,1)--cycle,gray(0.7)); \nfill((4,-1)--(4,0)--(5,0)--(5,-1)--cycle,gray(0.7)); \nfill((5,0)--(5,1)--(6,1)--(6,0)--cycle,gray(0.7)); \nfill((5,-2)--(5,-1)--(6,-1)--(6,-2)--cycle,gray(0.7)); \ndraw((3,-2)--(3,2)--(6,2)--(6,-2)--cycle,linewidth(0.7)); \ndraw((3,-1)--(6,-1),linewidth(0.7)); \ndraw((3,0)--(6,0),linewidth(0.7)); \ndraw((3,1)--(6,1),linewidth(0.7)); \ndraw((4,-2)--(4,2),linewidth(0.7)); \ndraw((5,-2)--(5,2),linewidth(0.7)); \nlabel("1",(3.5,-2),S); \nlabel("2",(4.5,-2),S); \nlabel("3",(5.5,-2),S); \nlabel("1",(3,-1.5),W); \nlabel("2",(3,-0.5),W); \nlabel("3",(3,0.5),W); \nlabel("4",(3,1.5),W); \n[/asy]")
    # parser.add_argument("--question", type=str, default="Jen enters a lottery by picking $4$ distinct numbers from $S=\{1,2,3,\cdots,9,10\}.$ $4$ numbers are randomly chosen from $S.$ She wins a prize if at least two of her numbers were $2$ of the randomly chosen numbers, and wins the grand prize if all four of her numbers were the randomly chosen numbers. The probability of her winning the grand prize given that she won a prize is $\\tfrac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.", help="Question to answer")
    # parser.add_argument("--question", type=str, default="Two identical rectangular crates are packed with cylindrical pipes, using different methods. Each pipe has diameter $10\text{ cm}.$ A side view of the first four rows of each of the two different methods of packing is shown below.\n\n[asy]\ndraw(circle((1,1),1),black+linewidth(1));\ndraw(circle((3,1),1),black+linewidth(1));\ndraw(circle((5,1),1),black+linewidth(1));\ndraw(circle((7,1),1),black+linewidth(1));\ndraw(circle((9,1),1),black+linewidth(1));\ndraw(circle((11,1),1),black+linewidth(1));\ndraw(circle((13,1),1),black+linewidth(1));\ndraw(circle((15,1),1),black+linewidth(1));\ndraw(circle((17,1),1),black+linewidth(1));\ndraw(circle((19,1),1),black+linewidth(1));\ndraw(circle((1,3),1),black+linewidth(1));\ndraw(circle((3,3),1),black+linewidth(1));\ndraw(circle((5,3),1),black+linewidth(1));\ndraw(circle((7,3),1),black+linewidth(1));\ndraw(circle((9,3),1),black+linewidth(1));\ndraw(circle((11,3),1),black+linewidth(1));\ndraw(circle((13,3),1),black+linewidth(1));\ndraw(circle((15,3),1),black+linewidth(1));\ndraw(circle((17,3),1),black+linewidth(1));\ndraw(circle((19,3),1),black+linewidth(1));\ndraw(circle((1,5),1),black+linewidth(1));\ndraw(circle((3,5),1),black+linewidth(1));\ndraw(circle((5,5),1),black+linewidth(1));\ndraw(circle((7,5),1),black+linewidth(1));\ndraw(circle((9,5),1),black+linewidth(1));\ndraw(circle((11,5),1),black+linewidth(1));\ndraw(circle((13,5),1),black+linewidth(1));\ndraw(circle((15,5),1),black+linewidth(1));\ndraw(circle((17,5),1),black+linewidth(1));\ndraw(circle((19,5),1),black+linewidth(1));\ndraw(circle((1,7),1),black+linewidth(1));\ndraw(circle((3,7),1),black+linewidth(1));\ndraw(circle((5,7),1),black+linewidth(1));\ndraw(circle((7,7),1),black+linewidth(1));\ndraw(circle((9,7),1),black+linewidth(1));\ndraw(circle((11,7),1),black+linewidth(1));\ndraw(circle((13,7),1),black+linewidth(1));\ndraw(circle((15,7),1),black+linewidth(1));\ndraw(circle((17,7),1),black+linewidth(1));\ndraw(circle((19,7),1),black+linewidth(1));\ndraw((0,15)--(0,0)--(20,0)--(20,15),black+linewidth(1));\ndot((10,9));\ndot((10,11));\ndot((10,13));\nlabel('Crate A',(10,0),S);\n[/asy]\n\n[asy]\ndraw(circle((1,1),1),black+linewidth(1));\ndraw(circle((3,1),1),black+linewidth(1));\ndraw(circle((5,1),1),black+linewidth(1));\ndraw(circle((7,1),1),black+linewidth(1));\ndraw(circle((9,1),1),black+linewidth(1));\ndraw(circle((11,1),1),black+linewidth(1));\ndraw(circle((13,1),1),black+linewidth(1));\ndraw(circle((15,1),1),black+linewidth(1));\ndraw(circle((17,1),1),black+linewidth(1));\ndraw(circle((19,1),1),black+linewidth(1));\ndraw(circle((2,2.75),1),black+linewidth(1));\ndraw(circle((4,2.75),1),black+linewidth(1));\ndraw(circle((6,2.75),1),black+linewidth(1));\ndraw(circle((8,2.75),1),black+linewidth(1));\ndraw(circle((10,2.75),1),black+linewidth(1));\ndraw(circle((12,2.75),1),black+linewidth(1));\ndraw(circle((14,2.75),1),black+linewidth(1));\ndraw(circle((16,2.75),1),black+linewidth(1));\ndraw(circle((18,2.75),1),black+linewidth(1));\ndraw(circle((1,4.5),1),black+linewidth(1));\ndraw(circle((3,4.5),1),black+linewidth(1));\ndraw(circle((5,4.5),1),black+linewidth(1));\ndraw(circle((7,4.5),1),black+linewidth(1));\ndraw(circle((9,4.5),1),black+linewidth(1));\ndraw(circle((11,4.5),1),black+linewidth(1));\ndraw(circle((13,4.5),1),black+linewidth(1));\ndraw(circle((15,4.5),1),black+linewidth(1));\ndraw(circle((17,4.5),1),black+linewidth(1));\ndraw(circle((19,4.5),1),black+linewidth(1));\ndraw(circle((2,6.25),1),black+linewidth(1));\ndraw(circle((4,6.25),1),black+linewidth(1));\ndraw(circle((6,6.25),1),black+linewidth(1));\ndraw(circle((8,6.25),1),black+linewidth(1));\ndraw(circle((10,6.25),1),black+linewidth(1));\ndraw(circle((12,6.25),1),black+linewidth(1));\ndraw(circle((14,6.25),1),black+linewidth(1));\ndraw(circle((16,6.25),1),black+linewidth(1));\ndraw(circle((18,6.25),1),black+linewidth(1));\ndraw((0,15)--(0,0)--(20,0)--(20,15),black+linewidth(1));\ndot((10,9));\ndot((10,11));\ndot((10,13));\nlabel('Crate B',(10,0),S);\n[/asy]\n\nThree pipes from Crate $B$ are shown. Determine the height, $h,$ of this pile of $3$ pipes.\n\n[asy]\ndraw(circle((10,10),10),black+linewidth(1));\ndraw(circle((30,10),10),black+linewidth(1));\ndraw(circle((20,27.5),10),black+linewidth(1));\ndraw((50,0)--(50,37.5),black+linewidth(1));\ndraw((49,0)--(51,0),black+linewidth(1));\ndraw((49,37.5)--(51,37.5),black+linewidth(1));\nlabel('$h$',(50,0)--(50,37.5),E);\n[/asy]")

    # This below gave highest offloading tendency from 1.5B on AIME24NoFigs
    parser.add_argument("--question", type=str, default="Alice chooses a set $A$ of positive integers. Then Bob lists all finite nonempty sets $B$ of positive integers with the property that the maximum element of $B$ belongs to $A$. Bob's list has 2024 sets. Find the sum of the elements of A.")
    # parser.add_argument("--question", type=str, default="How many r's in strawberrannabeRryberranaberry?")
    parser.add_argument("--test_logging", action="store_true",
                        help="Enable test_logging mode in the service request.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host where spec_service is running.")
    parser.add_argument("--service_port", type=int, default=5000,
                        help="Port for the spec_service Flask app.")
    parser.add_argument("--big_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
    parser.add_argument("--big_model_port", type=int, default=8000)
    parser.add_argument("--big_model_gpus", type=str, default="1,2")
    parser.add_argument("--small_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--small_model_port", type=int, default=8001)
    parser.add_argument("--small_model_gpus", type=str, default="0")
    ### Sequential scaling of big/small/logprob modes
    parser.add_argument("--sequential_scale", type=int, default=0)
    ### Modes, only 1 can be true ###
    parser.add_argument("--placeholder_mode", action="store_true")
    parser.add_argument("--spec_rewrite", action="store_true")
    parser.add_argument("--spec_reason", action="store_true")
    parser.add_argument("--random_switch", action="store_true")
    parser.add_argument("--logprob_subselect", action="store_true")
    parser.add_argument("--big_model_only", action="store_true")
    parser.add_argument("--small_model_only", action="store_true")
    ### End Of Modes, only 1 can be true ###
    ### LogProb Subselect Args ###
    parser.add_argument("--sgen", type=int, default=256)
    parser.add_argument("--stok", type=int, default=16)
    parser.add_argument("--sdecay", type=int, default=2)
    parser.add_argument("--ltok", type=int, default=16)
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
    parser.add_argument("--terminating_string", type=str, default=r"\\n Put your final answer within \boxed{}.")
    parser.add_argument("--terminate_on_exit", action="store_true",
                        help="Stop the vLLM servers on exit.")
    parser.add_argument("--spec_service_path", type=str, default="./spec_service.py",
                        help="Path to the spec_service.py script.")

    args = parser.parse_args()
    # python test_spec.py --test_logging --big_model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --big_model_gpus 0 --small_model_gpus 1 --small_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --logprob_subselect ---sgen 512 --stok 16 --sdecay 2 --ltok 32

    if args.max_iterations is None:
        args.max_iterations = 32768 // (args.stok * args.ltok)
    
    if sum([args.placeholder_mode, args.spec_rewrite, args.logprob_subselect, args.big_model_only, args.small_model_only, args.random_switch, args.spec_reason]) != 1:
        print("[test_spec] Exactly one of placeholder_mode, spec_rewrite, spec_reason, logprob_subselect, big_model_only, small_model_only, random_switch should be True.")
        sys.exit(1)

    # kill_cmd = "fuser -k -9 /dev/nvidia*"
    # subprocess.run(kill_cmd, shell=True)

    if ping_service(args.host, args.service_port):
        print("[test_spec] Service is already running; shutting it down.")
        shutdown_url = f"http://{args.host}:{args.service_port}/shutdown"
        try:
            requests.post(shutdown_url, timeout=5)
        except requests.exceptions.RequestException:
            pass

        start_time = time.time()
        while ping_service(args.host, args.service_port):
            if time.time() - start_time > 30:
                print("[test_spec] Could not shut down the service within 30s; exiting.")
                sys.exit(1)
            time.sleep(1)
        print("[test_spec] Old service has been shut down.")

    print("[test_spec] Launching spec_service.py...")
    cmd = [
            "python", args.spec_service_path,
            f"--big_model={args.big_model}",
            f"--big_model_port={args.big_model_port}",
            f"--big_model_gpus={args.big_model_gpus}",
            f"--small_model={args.small_model}",
            f"--small_model_port={args.small_model_port}",
            f"--small_model_gpus={args.small_model_gpus}",
            f"--thinking_n_ignore={args.thinking_n_ignore}",
            f"--drafting_n={args.drafting_n}",
            f"--bloat_tokens={args.bloat_tokens}",
            f"--max_tokens={args.max_tokens}",
            f"--terminating_string={args.terminating_string}",
            f"--sgen={args.sgen}",
            f"--stok={args.stok}",
            f"--sdecay={args.sdecay}",
            f"--ltok={args.ltok}",
            f"--lbound={args.lbound}",
            f"--max_iterations={args.max_iterations}",
            f"--switch_ratio={args.switch_ratio}",
            f"--switch_chunk={args.switch_chunk}",
            "--port", str(args.service_port),
            "--sequential_scale", str(args.sequential_scale)
        ]
    # Handle optional args as before
    if args.random_switch:
        cmd.append("--random_switch")
    if args.small_first:
        cmd.append("--small_first")
    if args.placeholder_mode:
        cmd.append("--placeholder_mode")
    if args.logprob_subselect:
        cmd.append("--logprob_subselect")
    if args.spec_rewrite:
        cmd.append("--spec_rewrite")
    if args.spec_reason:
        cmd.append("--spec_reason")
    if args.big_model_only:
        cmd.append("--big_model_only")
    if args.small_model_only:
        cmd.append("--small_model_only")
    if args.full_rewrite:
        cmd.append("--full_rewrite")
    if args.draft_propose_ignore_str:
        cmd.append("--draft_propose_ignore_str")
    if args.terminate_on_exit:
        cmd.append("--terminate_on_exit")

    proc = subprocess.Popen(cmd, env=os.environ.copy())

    print("[test_spec] Waiting for service to come up...")
    if not wait_for_service(args.host, args.service_port, timeout=300):
        print("[test_spec] Service did not come up in time. Exiting.")
        proc.terminate()
        sys.exit(1)

    print("[test_spec] Service is now up.")

    # Step 2: Send the question to /speculative_reason
    url = f"http://{args.host}:{args.service_port}/speculative_reason"
    payload = {
        "question": args.question,
        "thinking_n_ignore": args.thinking_n_ignore,
        "drafting_n": args.drafting_n,
        "full_rewrite": args.full_rewrite,
        "random_switch": args.random_switch,
        "small_first": args.small_first,
        "placeholder_mode": args.placeholder_mode,
        "logprob_subselect": args.logprob_subselect,
        "spec_rewrite": args.spec_rewrite,
        "spec_reason": args.spec_reason,
        "big_model_only": args.big_model_only,
        "small_model_only": args.small_model_only,
        "switch_ratio": args.switch_ratio,
        "switch_chunk": args.switch_chunk,
        "sgen": args.sgen,
        "stok": args.stok,
        "sdecay": args.sdecay,
        "ltok": args.ltok,
        "lbound": args.lbound,
        "max_iterations": args.max_iterations,
        "bloat_tokens": args.bloat_tokens,
        "max_tokens": args.max_tokens,
        "terminating_string": args.terminating_string,
        "test_logging": args.test_logging,
        "draft_propose_ignore_str": args.draft_propose_ignore_str,
        "sequential_scale": args.sequential_scale,
    }
    print(f"[test_spec] Sending request with payload = {payload}")
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    resp_json = resp.json()

    final_answer = resp_json.get("final_answer", "")
    usage_data = resp_json.get("usage_records", [])

    # Step 3: Print the final answer
    print("\n=== Final Answer ===")
    print(final_answer)

    # Step 4: Print usage data
    print_usage_table(usage_data)

    if args.terminate_on_exit:
        print("[test_spec] '--terminate_on_exit' was passed, so the service will shut down automatically when you Ctrl+C here.")
        # If you want to forcibly kill the service in code:
        # proc.terminate()
        # proc.wait()
        # But typically we let spec_service.py do that for us at exit.

if __name__ == "__main__":
    main()
