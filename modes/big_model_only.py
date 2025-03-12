# modes/big_model_only.py

from pprint import pprint
import os
import datetime


def run_bigmodel_flow(
    question,
    big_model,
    big_model_port,
    generate_text_vllm,
    max_tokens=1024,
    temperature=0.7
):
    """
    A baseline 'placeholder' flow: we just send a single request to the
    *big_model* and return it as a final answer, plus usage data.
    """
    usage_data = []

    if test_logging:
        draft_logs = "big_model_draft_logs"
        import os
        if not os.path.exists(draft_logs):
            os.makedirs(draft_logs)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        with open(f"{draft_logs}/arg_list_{timestamp}.txt", "w") as f:
            for name, value in locals().items():
                f.write(f"{name} = {value}\n")

        subfolder_path = f"{draft_logs}/{timestamp}"
        os.makedirs(subfolder_path, exist_ok=True)

    # Basic prompt
    if "｜" not in question:
        prompt = f"<｜begin▁of▁sentence｜><｜User｜>{question}{terminating_string}<｜Assistant｜>\n{model_think_prefix}"
    else:
        prompt = f"{question}{terminating_string}\n{model_think_prefix}"

    print("Sending request to big model")
    # Single big model request
    resp_json, latency = generate_text_vllm(
        prompt,
        port=big_model_port,
        temperature=temperature,
        max_tokens=max_tokens,
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
    return final_reply, usage_data
