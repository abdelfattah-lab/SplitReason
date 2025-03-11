# modes/placeholder.py

def run_placeholder_flow(
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

    # Basic prompt
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"

    # Single big model request
    resp_json, latency = generate_text_vllm(
        prompt,
        port=big_model_port,
        temperature=temperature,
        max_tokens=max_tokens,
        model=big_model
    )

    usage_dict = resp_json.get("usage", {})
    usage_data.append({
        "Model": big_model,
        "ThinkIter": "placeholder",
        "DraftVersion": 0,
        "PromptTokens": usage_dict.get("prompt_tokens", 0),
        "CompletionTokens": usage_dict.get("completion_tokens", 0),
        "Latency": latency
    })

    final_reply = resp_json["choices"][0]["text"]
    return final_reply, usage_data
