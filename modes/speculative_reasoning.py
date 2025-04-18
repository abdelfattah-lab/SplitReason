# modes/spec_rewrite.py
import os
import datetime
import re
import time
import traceback
import uuid

def get_bigmodel_mask(text, open_tag="<bigmodel>", close_tag="</bigmodel>"):
    mask = [0] * len(text)
    start_index = 0

    while True:
        open_pos = text.find(open_tag, start_index)
        if open_pos == -1:
            break  # no more openings

        close_pos = text.find(close_tag, open_pos + len(open_tag))
        if close_pos == -1:
            # If we can't find a close tag, mark until the end of the text
            for i in range(open_pos, len(text)):
                mask[i] = 1
            break
        else:
            # Mark the region from <bigmodel> ... </bigmodel>
            region_end = close_pos + len(close_tag)
            for i in range(open_pos, region_end):
                mask[i] = 1
            start_index = region_end

    return mask

def get_mask_mean_median_std(text, open_tag="<bigmodel>", close_tag="</bigmodel>"):
    mask = get_bigmodel_mask(text, open_tag, close_tag)
    coverage = 100.0 * sum(mask) / len(mask)
    mean = sum(mask) / len(mask)
    median = sorted(mask)[len(mask) // 2]
    std = (sum((x - mean) ** 2 for x in mask) / len(mask)) ** 0.5
    return coverage, mean, median, std


def write_op():
    with open("track_op.txt", "a") as f:
        f.write("x\n")

def write_ope():
    with open("track_op.txt", "a") as f:
        f.write("L\n")

def sanitize_question(question: str) -> str:
    terms_to_remove = ["<｜User｜>", "<｜Assistant｜>", "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<think>"]
    for term in terms_to_remove:
        question = question.replace(term, "")
    return question

def run_speculative_reasoning_flow(
    question: str,
    sgen: int,
    stok: int,
    sdecay: int,
    ltok: int,
    max_tokens: int,
    temperature: float,
    big_model: str,
    big_model_port: int,
    small_model: str,
    small_model_port: int,
    requests,
    batched_generate_text_vllm,
    batched_eval_logprob_vllm,
    batched_generate_text_with_tokens_vllm,
    terminating_string: str,
    test_logging: bool = False,
    lbound: int = 2,
    max_iterations: int = 100,
    sequential_scale=0,
    token_counter=None 
):
    """
    Full Speculative Reasoning flow, only for evaluation
    """
    # We'll fallback to the usage_data list that we accumulate
    usage_data: List[Dict[str, Any]] = []
    if test_logging:
        draft_logs = "random_switch_draft_logs"
        if not os.path.exists(draft_logs):
            os.makedirs(draft_logs)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        subfolder_path = os.path.join(draft_logs, timestamp)
        os.makedirs(subfolder_path, exist_ok=True)

    model_think_prefix = "<think>\n"
    model_think_suffix = "</think>"
    bigmodel_str = "You always use <bigmodel>...</bigmodel> to mark parts of the reasoning process that are important."
    question = sanitize_question(question)
    base_prompt = (
        f"<｜begin▁of▁sentence｜><｜User｜>{question}\n"
        f"{bigmodel_str}\n"
        f"{terminating_string}"
        f"<｜Assistant｜>\n"
        f"{model_think_prefix}"
    )
    print("*"*50)
    print(base_prompt)
    print("*"*50)
    current_text = base_prompt
    # Max per-chance bigmodel
    max_bigmodel_perchance = 256 # MPC
    # keep this small but not too small, it becomes the 'batch size' for batch small model generation on stop command
    numtok_bigmodel = 32
    # for every 16 possible generations, how many tokens to geenrate to check for </bigmodel>
    numsteps_smallmodel = 8
    MAX_PERM_TOKS = 16384
    finish_reason = 'uninitialized'

    start_time = time.time()
    while True:
        curr_token_count = token_counter(current_text)
        if curr_token_count > MAX_PERM_TOKS:
            print("Early stop premature result.")
            total_time = time.time() - start_time
            total_tokens = token_counter(current_text)
            time_per_tok = total_time / total_tokens
            uuid_ = str(uuid.uuid4())
            reason = "TOKEN_LENGTH"
            coverage, mean, median, std = get_mask_mean_median_std(current_text)
            # Save uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok to a file called "speculative_reasoning_benchmarks.csv"
            if not os.path.exists("speculative_reasoning_benchmarks.csv"):
                with open("speculative_reasoning_benchmarks.csv", "w") as f:
                    f.write("uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok,reason,coverage,mean,median,std\n")
            with open("speculative_reasoning_benchmarks.csv", "a") as f:
                f.write(f"{uuid_},{small_model},{numtok_bigmodel},{numsteps_smallmodel},{total_tokens},{total_time},{time_per_tok},{reason},{coverage},{mean},{median},{std}\n")
            write_ope()
            return current_text, usage_data
        # print("Small model invoked first/post-</bigmodel>")
        generation_resps, latency = batched_generate_text_vllm(
            prompts=[current_text],
            port=small_model_port,
            temperature=temperature,
            max_tokens=MAX_PERM_TOKS - curr_token_count,
            model=small_model,
            is_bigmodel_halting=True,
            requests=requests
            )
        if generation_resps is None:
            print(f"ERROR"); print(f"Traceback: {traceback.format_exc()}"); print("\n\n RETURNING EARLY OUTPUT \n\n")
            total_time = time.time() - start_time
            total_tokens = token_counter(current_text)
            time_per_tok = total_time / total_tokens
            uuid_ = str(uuid.uuid4())
            reason = "SMALL_MODEL_ERROR"
            coverage, mean, median, std = get_mask_mean_median_std(current_text)
            # Save uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok to a file called "speculative_reasoning_benchmarks.csv"
            if not os.path.exists("speculative_reasoning_benchmarks.csv"):
                with open("speculative_reasoning_benchmarks.csv", "w") as f:
                    f.write("uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok,reason,coverage,mean,median,std\n")
            with open("speculative_reasoning_benchmarks.csv", "a") as f:
                f.write(f"{uuid_},{small_model},{numtok_bigmodel},{numsteps_smallmodel},{total_tokens},{total_time},{time_per_tok},{reason},{coverage},{mean},{median},{std}\n")
            return current_text, usage_data

        current_text += generation_resps[0]['choices'][0]['text']
        finish_reason = generation_resps[0]['choices'][0]['finish_reason']

        if finish_reason == 'length':
            write_op()
            total_time = time.time() - start_time
            total_tokens = token_counter(current_text)
            time_per_tok = total_time / total_tokens
            uuid_ = str(uuid.uuid4())
            reason = "SMALL_MODEL_LENGTH"
            coverage, mean, median, std = get_mask_mean_median_std(current_text)
            # Save uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok to a file called "speculative_reasoning_benchmarks.csv"
            if not os.path.exists("speculative_reasoning_benchmarks.csv"):
                with open("speculative_reasoning_benchmarks.csv", "w") as f:
                    f.write("uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok,reason,coverage,mean,median,std\n")
            with open("speculative_reasoning_benchmarks.csv", "a") as f:
                f.write(f"{uuid_},{small_model},{numtok_bigmodel},{numsteps_smallmodel},{total_tokens},{total_time},{time_per_tok},{reason},{coverage},{mean},{median},{std}\n")
            return current_text, usage_data
        else:
            # Here, the big-model is invoked.
            # So, we need to do: BigGenerate --> SmallCheckForBigEnd loop
            # Till small model has immediately emmited </bigmodel> tag
            inloop_times = 0
            while True:
                inloop_times += 1
                curr_token_count = token_counter(current_text)
                if curr_token_count > MAX_PERM_TOKS:
                    print("Early stop premature result.")
                    total_time = time.time() - start_time
                    total_tokens = token_counter(current_text)
                    time_per_tok = total_time / total_tokens
                    uuid_ = str(uuid.uuid4())
                    reason = "TOKEN_LENGTH_INLOOP"
                    coverage, mean, median, std = get_mask_mean_median_std(current_text)
                    # Save uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok to a file called "speculative_reasoning_benchmarks.csv"
                    if not os.path.exists("speculative_reasoning_benchmarks.csv"):
                        with open("speculative_reasoning_benchmarks.csv", "w") as f:
                            f.write("uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok,reason,coverage,mean,median,std\n")
                    with open("speculative_reasoning_benchmarks.csv", "a") as f:
                        f.write(f"{uuid_},{small_model},{numtok_bigmodel},{numsteps_smallmodel},{total_tokens},{total_time},{time_per_tok},{reason},{coverage},{mean},{median},{std}\n")
                    write_ope()
                    return current_text, usage_data
                # Here, finish reason will mostly be 'length' because bigmodel doesnt use <bigmodel> tags
                # however, if it is 'stop', then it means it emitted End-of-Text, so we can stop
                # and return the current text
                # print("Bigmodel invoked post <bigmodel>")
                generation_resps, generation_tokens, latency = batched_generate_text_with_tokens_vllm(
                    prompts=[current_text.replace("<bigmodel>", "").replace("</bigmodel>", "").replace(bigmodel_str, "")],
                    port=big_model_port,
                    temperature=temperature,
                    max_tokens=numtok_bigmodel,
                    model=big_model,
                    requests=requests,
                    logprobs=1
                )
                if generation_resps is None:
                    print(f"ERROR"); print(f"Traceback: {traceback.format_exc()}"); print("\n\n RETURNING EARLY OUTPUT \n\n")
                    reason = "BIG_MODEL_ERROR"
                    coverage, mean, median, std = get_mask_mean_median_std(current_text)
                    # Save uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok to a file called "speculative_reasoning_benchmarks.csv"
                    if not os.path.exists("speculative_reasoning_benchmarks.csv"):
                        with open("speculative_reasoning_benchmarks.csv", "w") as f:
                            f.write("uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok,reason,coverage,mean,median,std\n")
                    with open("speculative_reasoning_benchmarks.csv", "a") as f:
                        f.write(f"{uuid_},{small_model},{numtok_bigmodel},{numsteps_smallmodel},{total_tokens},{total_time},{time_per_tok},{reason},{coverage},{mean},{median},{std}\n")
                    write_ope()
                    return current_text, usage_data
                finish_reason = generation_resps[0]['choices'][0]['finish_reason']
                if finish_reason != 'length': # If it isnt length, i think its EoT, so return it
                    current_text += generation_resps[0]['choices'][0]['text']
                    # import pdb; pdb.set_trace()
                    total_time = time.time() - start_time
                    total_tokens = token_counter(current_text)
                    time_per_tok = total_time / total_tokens
                    uuid_ = str(uuid.uuid4())
                    reason = "BIG_MODEL_EOT"
                    coverage, mean, median, std = get_mask_mean_median_std(current_text)
                    # Save uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok to a file called "speculative_reasoning_benchmarks.csv"
                    if not os.path.exists("speculative_reasoning_benchmarks.csv"):
                        with open("speculative_reasoning_benchmarks.csv", "w") as f:
                            f.write("uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok,reason,coverage,mean,median,std\n")
                    with open("speculative_reasoning_benchmarks.csv", "a") as f:
                        f.write(f"{uuid_},{small_model},{numtok_bigmodel},{numsteps_smallmodel},{total_tokens},{total_time},{time_per_tok},{reason},{coverage},{mean},{median},{std}\n")
                    write_op()
                    return current_text, usage_data
                if inloop_times > int(max_bigmodel_perchance // numtok_bigmodel): # 16 * 16 = 256 tokens generated by bigmodel, enough
                    current_text += generation_resps[0]['choices'][0]['text']
                    current_text += "</bigmodel>"
                    # break out of while loop
                    break
                tokens = generation_tokens[0]  # single-item prompt → one list of tokens
                # Construct incremental completions
                intermediate_completions = []
                for i in range(1, len(tokens) + 1):
                    temptext = current_text + "".join(tokens[:i])
                    intermediate_completions.append(temptext)
                # So now, we have a list of intermediate completions
                # Now we need to check each of these for the </bigmodel> tag
                # by generating 8 token continuations for each intermediate completion with small model
                # and checking if any response completion starts with "</big"
                # print("Small model checking for </bigmodel> tendency")
                small_completions = batched_generate_text_vllm(
                    prompts=intermediate_completions,
                    port=small_model_port,
                    temperature=temperature,
                    max_tokens=numsteps_smallmodel,
                    model=small_model,
                    requests=requests,
                )
                if small_completions[0] is None:
                    print(f"ERROR"); print(f"Traceback: {traceback.format_exc()}"); print("\n\n RETURNING EARLY OUTPUT \n\n")
                    write_ope()
                    total_time = time.time() - start_time
                    total_tokens = token_counter(current_text)
                    time_per_tok = total_time / total_tokens
                    uuid_ = str(uuid.uuid4())
                    reason = "SMALL_MODEL_ERROR"
                    coverage, mean, median, std = get_mask_mean_median_std(current_text)
                    # Save uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok to a file called "speculative_reasoning_benchmarks.csv"
                    if not os.path.exists("speculative_reasoning_benchmarks.csv"):
                        with open("speculative_reasoning_benchmarks.csv", "w") as f:
                            f.write("uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok,reason,coverage,mean,median,std\n")
                    with open("speculative_reasoning_benchmarks.csv", "a") as f:
                        f.write(f"{uuid_},{small_model},{numtok_bigmodel},{numsteps_smallmodel},{total_tokens},{total_time},{time_per_tok},{reason},{coverage},{mean},{median},{std}\n")
                    return intermediate_completions[-1], usage_data
                # Check if any small model continuation starts with the end tag
                bigmodel_job_done = False
                # Check if ANY of the next completions can invoke </bigmodel>
                for idx, resp in enumerate(small_completions[0]):
                    text = resp['choices'][0]['text']
                    if text.strip().__contains__("</big"):
                        # We have a match, so we can use that index intermediate completion and add </bigmodel>
                        pretext = text.split("</big")[0]
                        current_text = intermediate_completions[idx] + pretext + "</bigmodel>"
                        bigmodel_job_done = True
                        break
                # bigmodel job is done, now ask small model to continue from the </bigmodel> part onwards
                if bigmodel_job_done:
                    break
                # bigmodel job is NOT DONE. add the generation_resps text to current_text
                else:
                    current_text += generation_resps[0]['choices'][0]['text']
    total_time = time.time() - start_time
    total_tokens = token_counter(current_text)
    time_per_tok = total_time / total_tokens
    uuid_ = str(uuid.uuid4())
    reason = "ENDOF_LOOP"
    coverage, mean, median, std = get_mask_mean_median_std(current_text)
    # Save uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok to a file called "speculative_reasoning_benchmarks.csv"
    if not os.path.exists("speculative_reasoning_benchmarks.csv"):
        with open("speculative_reasoning_benchmarks.csv", "w") as f:
            f.write("uuid,small_model,numtok_bigmodel,numsteps_smallmodel,total_tokens,total_time,time_per_tok,reason,coverage,mean,median,std\n")
    with open("speculative_reasoning_benchmarks.csv", "a") as f:
        f.write(f"{uuid_},{small_model},{numtok_bigmodel},{numsteps_smallmodel},{total_tokens},{total_time},{time_per_tok},{reason},{coverage},{mean},{median},{std}\n")
    write_op()
    return current_text, usage_data
