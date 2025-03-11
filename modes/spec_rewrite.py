# modes/spec_rewrite.py

def run_speculative_reason_flow(
    question,
    test_logging,
    thinking_n_ignore,
    drafting_n,
    full_rewrite,
    temperature,
    max_tokens,
    terminating_string,
    draft_propose_ignore_str,
    small_first,
    big_model_port,
    big_model,
    small_model_port,
    small_model,
    bloat_tokens,
    generate_text_vllm,
    extract_cot,
    service_args
):
    """
    This function encapsulates the entire speculative reasoning flow
    that was originally inside speculative_reason().
    """
    # We'll fallback to the usage_data list that we accumulate
    usage_data = []

    # Potentially expand the question with "bloat tokens" if needed
    if bloat_tokens > 0:
        bloat_sentence = ("question to follow soon. " * 20)  # ~100 tokens
        question = bloat_sentence * (bloat_tokens // 100) + question

    # Basic prompt for user query
    base_prompt = f"<|user|>\n{question}\n<|assistant|>\n"

    # Depending on your models, define their prefix/suffix tokens
    if "simplescaling" in big_model:
        big_model_think_prefix = "<|im_start|>think\n"
        big_model_think_suffix = "<|im_start|>answer"
    elif "deepseek-ai" in big_model:
        big_model_think_prefix = "<think>\n"
        big_model_think_suffix = "\n</think>"
    else:
        # Fail out or handle differently
        raise ValueError("Unknown big model format.")

    if "simplescaling" in small_model:
        small_model_think_prefix = "<|im_start|>think\n"
        small_model_think_suffix = "<|im_start|>answer"
    elif "deepseek-ai" in small_model:
        small_model_think_prefix = "<think>\n"
        small_model_think_suffix = "\n</think>"
    else:
        # For models that do not have specialized tokens
        small_model_think_prefix = "\n"
        small_model_think_suffix = "\n"

    # We accumulate chain-of-thought in cot_accumulator
    cot_accumulator = ""
    wait_str = "\nWait"
    fallback_suffixes = ()

    print("\n\n Running with the following parameters: \n")
    print(f"thinking_n_ignore: {thinking_n_ignore}")
    print(f"drafting_n: {drafting_n}")
    print(f"full_rewrite: {full_rewrite}")
    print(f"terminating_string: {terminating_string}")
    print("\n\n")

    # ---------------------------
    # 1) The "thinking" iterations
    # ---------------------------
    for i in range(thinking_n_ignore):
        print("\n\n At thinking ignore iteration:", i)

        # Decide big vs small model for each iteration
        if i == 0 and small_first:
            model_port = small_model_port
            model_name = small_model
            model_think_prefix = small_model_think_prefix
            model_think_suffix = small_model_think_suffix
        else:
            model_port = big_model_port
            model_name = big_model
            model_think_prefix = big_model_think_prefix
            model_think_suffix = big_model_think_suffix

        # Prompt includes previously accumulated CoT
        if i == 0:
            iteration_prompt = base_prompt + model_think_prefix
        else:
            iteration_prompt = base_prompt + cot_accumulator

        # Generate from the chosen model
        raw_resp, latency = generate_text_vllm(
            iteration_prompt,
            port=model_port,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model_name
        )

        usage_dict = raw_resp.get("usage", {})
        usage_data.append({
            "Model": model_name,
            "ThinkIter": i + 1,
            "DraftVersion": 0,
            "PromptTokens": usage_dict.get("prompt_tokens", 0),
            "CompletionTokens": usage_dict.get("completion_tokens", 0),
            "Latency": latency
        })

        raw_reply = raw_resp["choices"][0]["text"]
        partial_cot = extract_cot(raw_reply, model_think_suffix, fallback_suffixes)

        # If test logging is enabled, write out debug logs
        if test_logging:
            draft_logs = "draft_logs"
            import os
            if not os.path.exists(draft_logs):
                os.makedirs(draft_logs)

            write_model_name = model_name.split("/")[-1]
            with open(f"{draft_logs}/{write_model_name}_iter{i+1}.txt", "w") as f:
                f.write("\n" + "-" * 80 + "\n" + "Iteration Prompt\n" + "-" * 80 + "\n")
                f.write(iteration_prompt)
                f.write("\n\n" + "-" * 80 + "\nRaw Reply\n" + "-" * 80 + "\n")
                f.write(raw_reply)
                f.write("\n\n" + "-" * 80 + "\nExtracted CoT\n" + "-" * 80 + "\n")
                f.write(partial_cot)

        # ---------------------------
        # 2) Drafting iterations (small model refinement)
        # ---------------------------
        if drafting_n > 0:
            drafts = []
            for d_i in range(drafting_n):
                print("\n\n At drafting iteration:", d_i)

                if full_rewrite:
                    prompt_for_draft = (
                        f"The question asked:\n{question}\n\n"
                        f"We currently have partial reasoning:\n\n"
                        f"{cot_accumulator} \t {partial_cot}\n"
                        f"{small_model_think_prefix}"
                        "I want to concisely refine the above reasoning, "
                        "preserving all the key steps / core reasoning."
                    )
                else:
                    if len(cot_accumulator) == 0:
                        cot_inject = "No Partial CoT yet"
                    else:
                        cot_inject = cot_accumulator

                    prompt_for_draft = (
                        f"The question asked:\n{question}\n\n"
                        f"Prior reasoning chain:\n"
                        f"{partial_cot}\n"
                        f"{small_model_think_prefix}"
                        "I want to refine the Partial Reasoning Trace, "
                        "keeping all the key steps / core reasoning."
                    )

                if draft_propose_ignore_str:
                    prompt_for_draft += (
                        "I must also conclude with a question that encourages deeper investigation "
                        "to make the steps more robust."
                        f"{small_model_think_suffix}"
                    )
                else:
                    prompt_for_draft += f"{small_model_think_suffix}"

                draft_resp, draft_latency = generate_text_vllm(
                    prompt_for_draft,
                    port=small_model_port,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=small_model
                )
                d_usage_dict = draft_resp.get("usage", {})
                usage_data.append({
                    "Model": small_model,
                    "ThinkIter": i + 1,
                    "DraftVersion": d_i + 1,
                    "PromptTokens": d_usage_dict.get("prompt_tokens", 0),
                    "CompletionTokens": d_usage_dict.get("completion_tokens", 0),
                    "Latency": draft_latency
                })

                raw_draft = draft_resp["choices"][0]["text"]
                drafts.append(raw_draft)

                if test_logging:
                    write_model_name = small_model.split("/")[-1]
                    with open(f"{draft_logs}/{write_model_name}_iter{i+1}_draft{d_i+1}.txt", "w") as f:
                        f.write("\n" + "-" * 80 + "\n" + "Prompt For Drafting\n" + "-" * 80 + "\n")
                        f.write(prompt_for_draft)
                        f.write("\n\n" + "-" * 80 + "\nRe-drafted Prompt\n" + "-" * 80 + "\n")
                        f.write(raw_draft)

            # Pick "best" draft by some trivial scoring (length, etc.)
            best_score = -1e9
            best_draft = drafts[0]
            for d in drafts:
                s = len(d)
                if s > best_score:
                    best_score = s
                    best_draft = d
            partial_cot = best_draft

        # Accumulate partial chain-of-thought
        if full_rewrite:
            cot_accumulator = partial_cot + wait_str
        else:
            cot_accumulator += partial_cot + wait_str

        if test_logging:
            with open(f"{draft_logs}/cot_step_{i+1}.txt", "w") as f:
                f.write(cot_accumulator)

    # -------------------------------------------
    # 3) Final answer from the big model
    # -------------------------------------------
    final_prompt = base_prompt + terminating_string + cot_accumulator
    final_resp, final_latency = generate_text_vllm(
        final_prompt,
        port=big_model_port,
        temperature=temperature,
        max_tokens=max_tokens,
        model=big_model
    )
    usage_dict = final_resp.get("usage", {})
    usage_data.append({
        "Model": big_model,
        "ThinkIter": "final",
        "DraftVersion": 0,
        "PromptTokens": usage_dict.get("prompt_tokens", 0),
        "CompletionTokens": usage_dict.get("completion_tokens", 0),
        "Latency": final_latency
    })

    final_reply = final_resp["choices"][0]["text"]

    return final_reply, usage_data
