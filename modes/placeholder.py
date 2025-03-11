# modes/placeholder.py

from pprint import pprint

def run_placeholder_flow(
    question,
    big_model,
    big_model_port,
    small_model,
    small_model_port,
    generate_text_vllm,
    max_tokens=1024,
    temperature=0.7
):
    """
    A baseline 'placeholder' flow: we just send a single request to the
    *big_model* and return it as a final answer, plus usage data.
    """
    usage_data = []

    # Basic prompt
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    print("Sending request to big model")
    # Single big model request
    resp_json, latency_big = generate_text_vllm(
        prompt,
        port=big_model_port,
        temperature=temperature,
        max_tokens=max_tokens,
        model=big_model
    )
    usage_dict_big = resp_json.get("usage", {})
    final_reply = resp_json["choices"][0]["text"]

    # Single big model request
    resp_json, latency_small = generate_text_vllm(
        prompt,
        port=small_model_port,
        temperature=temperature,
        max_tokens=max_tokens,
        model=small_model
    )

    usage_dict_small = resp_json.get("usage", {})

    usage_data.append({
        "Model": big_model,
        "ThinkIter": "placeholder",
        "DraftVersion": 0,
        "PromptTokens": usage_dict_big.get("prompt_tokens", 0),           # Always expect this item.
        "CompletionTokens": usage_dict_big.get("completion_tokens", 0),   # Always expect this item.
        "Latency": latency_big,                                           # Always expect this item.
        "ModelSmall": small_model,
        "PromptTokensSmall": usage_dict_small.get("prompt_tokens", 0),
        "CompletionTokensSmall": usage_dict_small.get("completion_tokens", 0),
        "LatencySmall": latency_small

    })

    pprint(usage_data)
    final_reply_small = resp_json["choices"][0]["text"]
    print("Final reply from small model:\n\n", final_reply_small)
    print("\n\nFinal reply from big model:\n\n", final_reply)
    return final_reply, usage_data
