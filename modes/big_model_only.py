# modes/big_model_only.py

from pprint import pprint
import os
import datetime

import time 
import uuid

benchfile = "specR_big.csv"

def sanitize_question(question: str) -> str:
    terms_to_remove = ["<｜User｜>", "<｜Assistant｜>", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<think>"]
    for term in terms_to_remove:
        question = question.replace(term, "")
    return question

def run_bigmodel_flow(
    question,
    big_model,
    big_model_port,
    generate_text_vllm,
    terminating_string: str,
    max_tokens=1024,
    temperature=0.7,
    sequential_scale=0,
    test_logging: bool = False,
    token_counter=None,
):
    """
    A baseline 'placeholder' flow: we just send a single request to the
    *big_model* and return it as a final answer, plus usage data.
    """
    usage_data = []

    model_think_prefix = "<think>\n"
    model_think_suffix = "</think>"

    if test_logging:
        draft_logs = "big_model_draft_logs"
        if not os.path.exists(draft_logs):
            os.makedirs(draft_logs)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        with open(f"{draft_logs}/arg_list_{timestamp}.txt", "w") as f:
            for name, value in locals().items():
                f.write(f"{name} = {value}\n")

        subfolder_path = f"{draft_logs}/{timestamp}"
        os.makedirs(subfolder_path, exist_ok=True)

    start_time = time.time()
    # Scaling is 0 indexed, dont ask me why lol
    question = sanitize_question(question)
    for sequential_iter in range(sequential_scale + 1):
        if sequential_iter == 0:
            if "｜" not in question:
                prompt = f"<｜begin▁of▁sentence｜><｜User｜>{question}{terminating_string}<｜Assistant｜>\n{model_think_prefix}"
            else:
                prompt = f"{question}{terminating_string}\n{model_think_prefix}"

        if test_logging:
            print("Sending request to big model")
        # Single big model request
        resp_json, latency = generate_text_vllm(
            prompt,
            port=big_model_port,
            temperature=temperature,
            max_tokens=8192,
            # max_tokens=16384,
            model=big_model
        )
        usage_dict = resp_json.get("usage", {})
        final_reply = resp_json["choices"][0]["text"]

        usage_data.append({
            "Model": big_model,
            "ThinkIter": "placeholder",
            "DraftVersion": 0,
            "PromptTokens": usage_dict.get("prompt_tokens", 0),           # Always expect this item.
            "CompletionTokens": usage_dict.get("completion_tokens", 0),   # Always expect this item.
            "Latency": latency,                                           # Always expect this item.

        })

        final_reply_big = resp_json["choices"][0]["text"]
        if sequential_scale > 0 and sequential_iter < sequential_scale - 1:
            # Add a '\nWait' to the final_reply and over-write prompt for the next iteration
            prompt = f"{prompt}{final_reply_big}\nWait"
    total_time = time.time() - start_time
    total_tokens = token_counter(final_reply) if token_counter else len(final_reply.split())
    time_per_tok = total_time / total_tokens if total_tokens > 0 else 0
    uuid_ = str(uuid.uuid4())
    if not os.path.exists(benchfile):
        with open(benchfile, "w") as f:
            f.write("uuid,big_model,sequential_scale,total_tokens,total_time,time_per_tok\n")
    with open(benchfile, "a") as f:
        f.write(f"{uuid_},{big_model},{sequential_scale},{total_tokens},{total_time},{time_per_tok}\n")
    return final_reply, usage_data
