# modes/big_model_only.py

from pprint import pprint
import os
import datetime

import time 
import uuid

benchfile = "specR_big.csv"

def run_bigmodel_flow(
    question,
    big_model,
    big_model_port,
    generate_text_vllm,
    terminating_string: str,
    max_tokens=1024,
    temperature=0.6,
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

    start_time = time.time()
    def _clean(t):  # strip special markers
        for s in ("<｜User｜>", "<｜Assistant｜>", "<｜begin▁of▁sentence｜>",
                  "<｜end▁of▁sentence｜>", "<think>"):
            t = t.replace(s, "")
        return t

    sequential_iter = 0 # Remove sequential-iter support
    if sequential_iter == 0:
        big_hint = ""
        term_str = "\n Put your final answer within \\boxed{}."
        cur = (f"<｜begin▁of▁sentence｜><｜User｜>{_clean(question)}\n"
            f"{big_hint}{term_str}<｜Assistant｜>\n<think>\n")
        prompt = cur
        
    resp_json, latency = generate_text_vllm(
        prompt,
        port=big_model_port,
        temperature=temperature,
        max_tokens=8192,
        model=big_model
    )
    final_reply = resp_json["choices"][0]["text"]
    # final_reply = f"{prompt}{final_reply}"
    final_reply = f"{final_reply}"
    total_time = time.time() - start_time
    total_tokens = token_counter(final_reply) if token_counter else len(final_reply.split())
    time_per_tok = total_time / total_tokens if total_tokens > 0 else 0
    uuid_ = str(uuid.uuid4())
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        if not os.path.exists(benchfile):
            with open(benchfile, "w") as f:
                f.write(
                    "uuid,big_model,sequential_scale,total_tokens,"
                    "total_time,time_per_tok,datetime\n"
                )
        with open(benchfile, "a") as f:
            f.write(
                f"{uuid_},{big_model},{sequential_scale},"
                f"{total_tokens},{total_time},{time_per_tok},{current_time}\n"
            )
    except Exception as e:
        print(f"Error writing to file: {e}")
        print("Please check if the file path is correct and if you have write permissions.")
        pass
    return final_reply, usage_data
