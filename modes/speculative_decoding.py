def run_speculative_decoding_flow(
    question: str,
    big_model: str,
    big_model_port: int,
    generate_text_vllm,
    max_tokens: int,
    temperature: float,
    test_logging: bool = False,
):
    # 1) Call the vLLM OpenAI‐compatible server
    resp_json, latency = generate_text_vllm(
        question,
        port=big_model_port,
        temperature=temperature,
        max_tokens=max_tokens,
        model=big_model
    )

    print(resp_json)

    # 2) Extract what we already had
    usage = resp_json.get("usage", {})
    final_reply = resp_json["choices"][0]["text"]

    # 3) Now pull out the extra stats
    spec_stats = resp_json.get("speculative_decoding_stats", {})

    # 4) Build the same usage record
    usage_data = [{
        "Model": big_model,
        "ThinkIter": "spec_decoding",
        "DraftVersion": 0,
        "PromptTokens": usage.get("prompt_tokens", 0),
        "CompletionTokens": usage.get("completion_tokens", 0),
        "Latency": latency,
    }]

    # 5) Return the new third argument
    return final_reply, usage_data, spec_stats