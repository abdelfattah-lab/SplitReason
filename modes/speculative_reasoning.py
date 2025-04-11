# modes/spec_rewrite.py
import os
import datetime
import re
import time
import traceback

def write_op():
    with open("track_op.txt", "a") as f:
        f.write("x\n")

def write_ope():
    with open("track_op.txt", "a") as f:
        f.write("L\n")

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
    base_prompt = (
        f"<｜begin▁of▁sentence｜><｜User｜>{question}\n"
        f"{terminating_string} "
        f"{bigmodel_str}"
        f"<｜Assistant｜>\n"
        f"{model_think_prefix}"
    )
    current_text = base_prompt
    # keep this small but not too small, it becomes the 'batch size' for batch small model generation on stop command
    numtok_bigmodel = 32
    # for every 16 possible generations, how many tokens to geenrate to check for </bigmodel>
    numsteps_smallmodel = 8
    finish_reason = 'uninitialized'

    while True:
        curr_token_count = token_counter(current_text)
        if curr_token_count > 16000:
            print("Early stop premature result.")
            write_ope()
            return current_text, usage_data
        # print("Small model invoked first/post-</bigmodel>")
        generation_resps, latency = batched_generate_text_vllm(
            prompts=[current_text],
            port=small_model_port,
            temperature=temperature,
            max_tokens=16000 - curr_token_count,
            model=small_model,
            requests=requests
            )
        if generation_resps is None:
            print(f"ERROR"); print(f"Traceback: {traceback.format_exc()}"); print("\n\n RETURNING EARLY OUTPUT \n\n")
            return current_text, usage_data
        current_text += generation_resps[0]['choices'][0]['text']
        finish_reason = generation_resps[0]['choices'][0]['finish_reason']

        if finish_reason == 'length':
            current_text += generation_resps[0]['choices'][0]['text']
            write_op()
            return current_text, usage_data
        else:
            # Here, the big-model is invoked.
            # So, we need to do: BigGenerate --> SmallCheckForBigEnd loop
            # Till small model has immediately emmited </bigmodel> tag
            while True:
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
                    write_ope()
                    return current_text, usage_data
                finish_reason = generation_resps[0]['choices'][0]['finish_reason']
                if finish_reason != 'length': # If it isnt length, i think its EoT, so return it
                    current_text += generation_resps[0]['choices'][0]['text']
                    # import pdb; pdb.set_trace()
                    write_op()
                    return current_text, usage_data

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
    write_op()
    return current_text, usage_data
