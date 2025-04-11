# modes/small_model_only.py

from pprint import pprint
import os
import datetime


def run_smallmodel_flow(
    question,
    small_model,
    small_model_port,
    generate_text_vllm,
    terminating_string: str,
    max_tokens=1024,
    temperature=0.7,
    test_logging: bool = False,
    sequential_scale=0,
    token_counter=None
):
    """
    A baseline 'placeholder' flow: we just send a single request to the
    *small_model* and return it as a final answer, plus usage data.
    """
    usage_data = []

    model_think_prefix = "<think>\n"
    model_think_suffix = "</think>"

    if test_logging:
        draft_logs = "small_model_draft_logs"
        import os
        if not os.path.exists(draft_logs):
            os.makedirs(draft_logs)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        with open(f"{draft_logs}/arg_list_{timestamp}.txt", "w") as f:
            for name, value in locals().items():
                f.write(f"{name} = {value}\n")

        subfolder_path = f"{draft_logs}/{timestamp}"
        os.makedirs(subfolder_path, exist_ok=True)

    # Scaling is 0 indexed, dont ask me why lol
    for sequential_iter in range(sequential_scale + 1):
        if sequential_iter == 0:
            # Basic prompt
            if "｜" not in question:
                prompt = f"<｜begin▁of▁sentence｜><｜User｜>{question}{terminating_string}<｜Assistant｜>\n{model_think_prefix}"
            else:
                prompt = f"{question}{terminating_string}\n{model_think_prefix}"

        if test_logging:
            print("Sending request to small model")
        # Single big model request
        resp_json, latency = generate_text_vllm(
            prompt,
            port=small_model_port,
            temperature=temperature,
            max_tokens=max_tokens,
            model=small_model
        )
        usage_dict = resp_json.get("usage", {})
        final_reply = resp_json["choices"][0]["text"]
        usage_data.append({
            "Model": small_model,
            "ThinkIter": "placeholder",
            "DraftVersion": 0,
            "PromptTokens": usage_dict.get("prompt_tokens", 0),           # Always expect this item.
            "CompletionTokens": usage_dict.get("completion_tokens", 0),   # Always expect this item.
            "Latency": latency,                                           # Always expect this item.

        })

        final_reply_small = resp_json["choices"][0]["text"]
        if sequential_scale > 0 and sequential_iter < sequential_scale:
            # Add a '\nWait' to the final_reply_small and over-write prompt for the next iteration
            prompt = f"{prompt}{final_reply_small}\nWait"  

    return final_reply_small, usage_data
