import time
import asyncio, datetime as _dt, json, os, time, traceback, uuid
from typing import Any, Dict, List, Tuple
import datetime

def run_speculative_decoding_flow(
    question: str,
    big_model: str,
    big_model_port: int,
    generate_text_vllm,
    max_tokens: int,
    temperature: float,
    test_logging: bool = False,
    token_counter=None
):
    start = time.time()
    # NOTE: calls V0 server, since V1 does not have specdec
    resp_json, latency, metric = generate_text_vllm(
        question,
        port=big_model_port,
        temperature=temperature,
        max_tokens=max_tokens,
        model=big_model,
        speculative_decoding=True # NOTE: custom parameter in generate_text_vllm
    )

    usage = resp_json.get("usage", {})
    final_reply = resp_json["choices"][0]["text"]

    usage_data = [{
        "Model":          big_model,
        "ThinkIter":      "spec_decoding",
        "DraftVersion":   0,
        "PromptTokens":   usage.get("prompt_tokens", 0),
        "CompletionTokens": usage.get("completion_tokens", 0),
        "AcceptedTokens":   metric["accepted_tokens"], # NOTE: these values are wrong, which are fetched directly from /metrics.
        "DraftTokens":      metric["draft_tokens"],
        "EmittedTokens":    metric["emitted_tokens"],
        "AcceptanceRate":   metric["acceptance_rate"],
        "Efficiency":       metric["efficiency"],
        "Latency":         latency,
    }]

    tot = time.time() - start
    csv = "specdecode_benchmark.csv"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_tokens = token_counter(final_reply) if token_counter else len(final_reply.split())
    time_per_tok = tot / total_tokens if total_tokens > 0 else 0
    try:
        if not os.path.exists(csv):
            with open(csv, "w") as f:
                f.write(
                    "uuid,big_model,total_tokens,total_time,time_per_tok,datetime\n"
                )
        with open(csv, "a") as f:
            f.write(
                f"{uuid.uuid4()},{big_model},{total_tokens},{tot},{time_per_tok},{now}\n"
            )
    except Exception as e:
        print(f"Error writing to file: {e}")
        print("Please check if the file path is correct and if you have write permissions.")
        pass
    return final_reply, usage_data