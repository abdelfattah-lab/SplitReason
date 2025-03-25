import traceback
from typing import List, Tuple
import random
from tqdm import tqdm
import os
import datetime
import time
from typing import List, Tuple, Dict, Any, Optional, Union
# final_reply, usage_data = run_random_switch_flow(
    #     question=question,
    #     test_logging=test_logging,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    #     terminating_string=terminating_string,
    #     big_model_port=service_args.big_model_port,
    #     big_model=service_args.big_model,
    #     small_model_port=service_args.small_model_port,
    #     small_model=service_args.small_model,
    #     batched_generate_text_vllm=batched_generate_text_vllm,
    #     batched_eval_logprob_vllm=batched_eval_logprob_vllm,
    #     switch_ratio=switch_ratio,
    #     switch_chunk=switch_chunk,
# )

def run_random_switch_flow(
    question: str,
    test_logging: bool,
    temperature: float,
    max_tokens: int,
    terminating_string: str,
    big_model_port: int,
    big_model: str,
    small_model_port: int,
    small_model: str,
    batched_generate_text_vllm,  # function for batched text generation
    batched_eval_logprob_vllm,   # (not used in this example, but included for consistency)
    switch_ratio: int,
    switch_chunk: int,
    requests,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Randomly switch between a small model and a big model when generating text.

    Args:
        question (str): The user question or prompt.
        test_logging (bool): Whether to log intermediate results to disk.
        temperature (float): Generation temperature for sampling.
        max_tokens (int): Maximum total tokens to generate before stopping.
        terminating_string (str): A delimiter string that indicates user query end.
        big_model_port (int): Port for the big model's service.
        big_model (str): The name/path of the big model.
        small_model_port (int): Port for the small model's service.
        small_model (str): The name/path of the small model.
        batched_generate_text_vllm (callable): A batched text generation function.
        batched_eval_logprob_vllm (callable): A batched logprob evaluation function (not used here).
        switch_ratio (int): An integer controlling the probability of choosing the big model.
                            Probability of choosing big model = 1/(switch_ratio + 1).
                            Probability of choosing small model = switch_ratio/(switch_ratio + 1).
        switch_chunk (int): Number of tokens to request each time we generate from a model.

    Returns:
        Tuple[str, List[Dict[str, Any]]]:
            final_prompt (str): The final text after random switching generation.
            usage_data (List[Dict[str, Any]]): A list of usage records (empty or minimal for now).
    """
    # Prepare test logging directory if needed
    if test_logging:
        draft_logs = "random_switch_draft_logs"
        if not os.path.exists(draft_logs):
            os.makedirs(draft_logs)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        subfolder_path = os.path.join(draft_logs, timestamp)
        os.makedirs(subfolder_path, exist_ok=True)

    # Probability of choosing the big model on each step
    # According to your spec:
    #   if switch_ratio = 1 => p_big = 1/(1+1) = 1/2 => 50:50
    #   if switch_ratio = 2 => p_big = 1/(2+1) = 1/3 => 33:66
    p_big = 1.0 / (switch_ratio + 1.0)

    # We'll track usage info here if needed
    usage_data: List[Dict[str, Any]] = []

    # We can prefix the prompt with something similar to your reference code
    model_think_prefix = "<think>\n"
    model_think_suffix = "</think>"

    if "｜" not in question:
        base_prompt = (
            f"<｜begin▁of▁sentence｜><｜User｜>{question}{terminating_string}<｜Assistant｜>\n{model_think_prefix}"
        )
    else:
        base_prompt = f"{question}{terminating_string}\n{model_think_prefix}"

    current_text = base_prompt
    tokens_generated = 0

    # Main generation loop
    while tokens_generated < max_tokens:
        # Decide which model to use (big or small) on this step
        use_big_model = (random.random() < p_big)
        chosen_model = big_model if use_big_model else small_model
        chosen_port = big_model_port if use_big_model else small_model_port

        # We'll request exactly switch_chunk tokens from the chosen model
        try:
            # We only have one prompt in the batch, so pass [current_text]
            generation_resps, latency = batched_generate_text_vllm(
                prompts=[current_text],
                port=chosen_port,
                temperature=temperature,
                max_tokens=switch_chunk,
                model=chosen_model,
                requests=requests  # you can pass your own requests/session object if needed
            )
        except Exception as e:
            print(f"ERROR while generating from {chosen_model}: {e}")
            print("Traceback:", traceback.format_exc())
            # Return early with what we have so far
            return current_text, usage_data

        partial_resp = generation_resps[0]["choices"][0]
        partial_text = partial_resp["text"]
        finish_reason = partial_resp["finish_reason"]  # e.g. 'stop', 'length', 'None'

        # Append the newly generated text to our current_text
        current_text += partial_text
        tokens_generated += switch_chunk

        # Optionally log the intermediate step
        if test_logging:
            # For demonstration, let's just store each step in a file
            log_filename = os.path.join(subfolder_path, f"step_{tokens_generated}.txt")
            with open(log_filename, "w", encoding="utf-8") as f:
                f.write(f"Model used: {chosen_model}\n")
                f.write(f"Finish reason: {finish_reason}\n")
                f.write(f"Partial text:\n{partial_text}\n")
                f.write("-" * 70 + "\n")
                f.write(f"Current full text so far:\n{current_text}\n")

        # We check if the model indicated an end of sentence or any finishing
        # condition. You can define your own condition more precisely:
        if finish_reason == "stop":
            break

        # Also break if we've reached the token limit
        if tokens_generated >= max_tokens:
            break

    final_prompt = current_text

    if test_logging:
        final_log_path = os.path.join(subfolder_path, "final_text.txt")
        with open(final_log_path, "w", encoding="utf-8") as f:
            f.write(final_prompt)

    # Return final text and usage data
    return final_prompt, usage_data