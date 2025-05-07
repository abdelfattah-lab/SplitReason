#!/usr/bin/env python3

import argparse
import os
import time
import subprocess
import sys
import requests
from tabulate import tabulate
import re

def ping_service(host: str, port: int, timeout: float = 3.0) -> bool:
    """Check if /ping endpoint returns 200 OK."""
    url = f"http://{host}:{port}/ping"
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False

def wait_for_service(host: str, port: int, timeout: float = 300.0) -> bool:
    """Wait up to `timeout` seconds for the /ping to succeed."""
    start = time.time()
    while time.time() - start < timeout:
        if ping_service(host, port):
            return True
        time.sleep(1)
    return False

def print_usage_table(usage_data):
    rows = []
    for ent in usage_data:
        model    = ent.get("Model", "")
        think    = ent.get("ThinkIter", "")
        ver      = ent.get("DraftVersion", "")
        p        = ent.get("PromptTokens", 0)
        c        = ent.get("CompletionTokens", 0)
        acc      = ent.get("AcceptedTokens", 0) # NOTE: these values are wrong, which are fetched directly from /metrics.
        draft    = ent.get("DraftTokens", 0)
        emit     = ent.get("EmittedTokens", 0)
        acc_rate = ent.get("AcceptanceRate", 0.0)
        eff      = ent.get("Efficiency", 0.0)
        lat      = ent.get("Latency", 0.0)
        per_tok  = lat / c if c else 0.0

        rows.append([
            model, think, ver, p, c,
            acc, draft, emit,
            round(acc_rate, 4), round(eff, 4),
            round(lat, 2),    round(per_tok, 4)
        ])

    print("\n=== Token Usage Summary ===")
    print(tabulate(
        rows,
        headers=[
            "Model","ThinkIter","DraftVer","PromptToks","CompleteToks",
            "Accepted","Draft","Emitted",
            "AccRate","Efficiency",
            "Latency(s)","Lat(s)/Tok"
        ],
        tablefmt="grid"
    ))
    print()


def main():
    parser = argparse.ArgumentParser(description="Run spec_service.py in speculative‑decoding mode and test it once.")
    parser.add_argument("--question", type=str, default="Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$.",
                        help="The prompt/question to send.")
    parser.add_argument("--test_logging", action="store_true",
                        help="Enable test‑logging in the service.")
    parser.add_argument("--terminate_on_exit", action="store_true",
                        help="Tell spec_service to shut down its vLLM servers on exit.")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Host where spec_service is running.")
    parser.add_argument("--service_port", type=int, default=5000,
                        help="Port for the spec_service Flask app.")
    parser.add_argument("--big_model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--big_model_port", type=int, default=8000)
    parser.add_argument("--big_model_gpus", type=str, default="0")
    parser.add_argument("--speculative_config", type=str, required=True,
                        help='JSON string for speculative_config, e.g. \'{"model":"<big>","num_speculative_tokens":5,"draft_tensor_parallel_size":1}\'.')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--enforce_eager", action="store_true",
                        help="Pass --enforce-eager to the vLLM server.")
    parser.add_argument("--spec_service_path", type=str, default="./spec_service.py",
                        help="Path to your spec_service.py entrypoint.")
    args = parser.parse_args()

    # 1) If a previous service is up, shut it down
    if ping_service(args.host, args.service_port):
        print("[test] Shutting down old service…")
        try:
            requests.post(f"http://{args.host}:{args.service_port}/shutdown", timeout=5)
        except:
            pass
        if not wait_for_service(args.host, args.service_port, timeout=30.0):
            # wait_for_service returns True if it comes up, so invert
            print("[test] Waiting for old service to die…")
            start = time.time()
            while ping_service(args.host, args.service_port):
                if time.time() - start > 30:
                    print("[test] Old service didn’t shut down in time; exiting.")
                    sys.exit(1)
                time.sleep(1)

    # 2) Launch spec_service.py in speculative‑decoding mode
    cmd = [
        "python", args.spec_service_path,
        f"--big_model={args.big_model}",
        f"--big_model_port={args.big_model_port}",
        f"--big_model_gpus={args.big_model_gpus}",
        "--spec_decoding",
        f"--speculative_config={args.speculative_config}",
        f"--seed={args.seed}",
        f"--gpu_memory_utilization={args.gpu_memory_utilization}",
        f"--max_model_len={args.max_model_len}",
    ]
    if args.enforce_eager:
        cmd.append("--enforce_eager")
    if args.test_logging:
        cmd.append("--test_logging")
    if args.terminate_on_exit:
        cmd.append("--terminate_on_exit")

    print("[test] Launching speculative‑decoding service…")
    proc = subprocess.Popen(cmd, env=os.environ.copy())

    # 3) Wait for it
    print(f"[test] Waiting for service at http://{args.host}:{args.service_port}/ping …")
    if not wait_for_service(args.host, args.service_port, timeout=300.0):
        print("[test] Service never came up; exiting.")
        proc.terminate()
        sys.exit(1)
    print("[test] Service is up.")

    # 4) Send our single question
    payload = {
        "question": args.question,
        "spec_decoding": True,
        "test_logging": args.test_logging,
        "max_tokens": 2048,
    }
    print(f"[test] Sending payload: {payload}")
    resp = requests.post(f"http://{args.host}:{args.service_port}/speculative_reason", json=payload)
    resp.raise_for_status()
    result = resp.json()

    # 5) Display
    print("\n=== Final Answer ===")
    print(result.get("final_answer", "").strip())
    print_usage_table(result.get("usage_records", []))

    if args.terminate_on_exit:
        print("[test] Service will shut down its vLLM servers on exit.")

if __name__ == "__main__":
    main()
