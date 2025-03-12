import traceback
from typing import List, Tuple
import random
from tqdm import tqdm
import os
import datetime
import time

# python test_spec.py  --logprob_subselect --stok 256 --sgen 32 --sdecay 2 --ltok 16 --lbound 2

### CAN WE DO SLIDING WINDOW WITH QUESTION ANCHOR TOKENS FOR LOGPROB EVAL ON LARGE MODELS


def eval_logprob_vllm(
    text_batch: List[str],
    big_model_port: int,
    big_model: str,
    requests,
    temperature: float = 0.0,
    max_tokens: int = 0,
    logprobs: int = 1
) -> Tuple[List[float], List[str]]:
    """
    Evaluate the total log-likelihood of each string in text_batch using
    the big model. Returns (scores, generated_texts).
    - `scores`: a list of total log-likelihood scores (higher is better).
    - `generated_texts`: the text *generated* by the model (though for
       a pure logprob evaluation, you typically set max_tokens=0 or 1).
    """
    scores = []
    gentexts = []
    url = f"http://localhost:{big_model_port}/v1/completions"
    big_model_gen_latencies = []

    for text in text_batch:
        try:
            payload = {
                "model": big_model,
                "prompt": text,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "logprobs": logprobs,
            }
            start_time = time.time()
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            end_time = time.time()
            big_model_gen_latencies.append(end_time - start_time)
            resp_json = resp.json()

            generated_text = resp_json["choices"][0]["text"]
            choice = resp_json["choices"][0]
            logprob_info = choice.get("logprobs")

            if not logprob_info:
                scores.append(float("-inf"))
                gentexts.append(generated_text)
                continue

            token_logprobs = logprob_info.get("token_logprobs", [])
            # Filter out None (sometimes found in the returned array)
            token_logprobs = [p for p in token_logprobs if p is not None]
            # prompt_ll = sum(token_logprobs) if len(token_logprobs) > 0 else 0.0
            if len(token_logprobs) > 0:
                prompt_ll =  sum(token_logprobs) / len(token_logprobs)
            else:
                prompt_ll = 0.0
            scores.append(prompt_ll)
            gentexts.append(generated_text)

        except Exception as e:
            import pdb; pdb.set_trace()
            traceback.print_exc()
            print("\n\n[WARNING]: Using Random Log-Prob Assignment, "
                  f"failed to get logprobs for text: {text}\n\n")
            scores.append(random.random())
            gentexts.append("")
    try:
        avg_latency = sum(big_model_gen_latencies) / len(big_model_gen_latencies)
    except:
        import pdb; pdb.set_trace()
    num_requests = len(text_batch)
    return scores, gentexts, avg_latency, num_requests


def run_logprob_subselect_flow(
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
    generate_text_vllm,
    terminating_string: str,
    test_logging: bool = False,
    lbound: int = 2,
    max_iterations: int = 100
):
    """
    A "logprob subselect" approach with two key modifications:
      1) We require `ltok > 0` and always let the large model generate
         more tokens after we subselect the best expansions.
      2) We have a lower bound (`lbound`) on the number of candidates.
         If the beam shrinks to <= lbound, we check how many contain the
         closing tag </think>. If a majority do, we finalize. Otherwise,
         we "double" the beam using the small model and continue.

    Arguments:
      - question: The user prompt to answer.
      - sgen: Initial number of variants (beam size) we generate in parallel.
      - stok: Number of tokens to generate from the small model each iteration.
      - sdecay: Factor by which we reduce the number of candidates after each iteration
                (n_keep = len(scores) // sdecay).
      - ltok: Number of tokens to generate from the big model each iteration (must be > 0).
      - max_tokens: Overall maximum tokens for final generation.
      - temperature: Temperature for the sampling calls.
      - big_model, small_model: Model names for vLLM's inference.
      - big_model_port, small_model_port: HTTP ports for each vLLM server.
      - requests: The `requests` library (or a compatible mock).
      - generate_text_vllm: A helper function to call vLLM (like in your snippet).
      - lbound: The lower bound on the number of candidate branches. If we reach
                <= lbound, we check for `</think>` in the majority of them.
      - max_iterations: A safeguard to avoid infinite loops.

    Returns:
      final_answer, usage_data
    """

    if test_logging:
        draft_logs = "logprob_subselect_draft_logs"
        import os
        if not os.path.exists(draft_logs):
            os.makedirs(draft_logs)

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        with open(f"{draft_logs}/arg_list_{timestamp}.txt", "w") as f:
            for name, value in locals().items():
                f.write(f"{name} = {value}\n")

        subfolder_path = f"{draft_logs}/{timestamp}"
        os.makedirs(subfolder_path, exist_ok=True)

    # 1) We enforce ltok > 0
    if ltok <= 0:
        raise ValueError("ltok must be > 0 for this approach.")

    usage_data = []

    model_think_prefix = "<think>\n"
    model_think_suffix = "</think>"

    # Start with a single shared prompt
    base_prompt = f"<user>\n{question}. {terminating_string}\n<assistant>\n{model_think_prefix}"

    # For the first step, replicate that prompt sgen times
    prompt_batch = [base_prompt for _ in range(sgen)]

    iteration_count = 0

    small_model_generations = {}
    while len(prompt_batch) > 1 and iteration_count < max_iterations:
        iteration_count += 1
        model_generation_latencies = {"small_model": [], "big_model": []}

        small_model_generations[iteration_count] = []
        new_candidates = []
        for prompt_text in prompt_batch:
            partial_resp, latency = generate_text_vllm(
                prompt_text,
                port=small_model_port,
                temperature=temperature,
                max_tokens=stok,
                model=small_model,
            )
            raw_reply = partial_resp["choices"][0]["text"]
            small_model_generations[iteration_count].append(raw_reply)
            new_candidates.append(prompt_text + raw_reply)
            model_generation_latencies["small_model"].append(latency)

        scores, big_generations, avg_latency, num_req_logprobs = eval_logprob_vllm(
            new_candidates,
            big_model_port=big_model_port,
            big_model=big_model,
            requests=requests,
            temperature=0.0,
            max_tokens=ltok,  # let the big model actually generate tokens
            logprobs=1
        )

        if test_logging:
            for i, (cand, score, big_gen) in enumerate(zip(new_candidates, scores, big_generations)):
                print(f"--- Candidate {i+1} \t \t Score: {score} ---")
                print(small_model_generations[iteration_count][i])
                print(f"--- BigModel Continuation {i+1} ---")
                print(big_gen)
                print("\n\n")
            sort_by_score = sorted(zip(new_candidates, scores, big_generations), key=lambda x: x[1], reverse=True)
            for i, (cand, score, big_gen) in enumerate(sort_by_score):
                mode = "w" if i == 0 else "a"
                write_model_name = big_model.split("/")[-1]
                with open(f"{subfolder_path}/{write_model_name}_iter{iteration_count}.txt", mode) as f:
                    f.write("\n" + "-" * 80 + "\n" + "Candidate\t \t Score: " + str(score) + "\n" + "-" * 80 + "\n")
                    f.write(small_model_generations[iteration_count][i])
                    f.write("\n" + "-" * 80 + "\n" + "BigModel Continuation\n" + "-" * 80 + "\n")
                    f.write(big_gen)
                    f.write("\n\n")

        # Combine each candidate with the big model's new text:
        combined_candidates = []
        for i, cand in enumerate(new_candidates):
            combined_candidates.append(cand + big_generations[i])

        # Now rank by the log-likelihood score:
        n_keep = max(1, len(scores) // sdecay)
        scored_candidates = list(zip(combined_candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        prompt_batch = [x[0] for x in scored_candidates[:n_keep]]


        # --- Step 5) If we are now at or below lbound, check majority for </think>
        if len(prompt_batch) <= lbound:
            # Count how many contain '</think>'
            n_with_think = sum(1 for cand in prompt_batch if model_think_suffix in cand)
            if n_with_think > len(prompt_batch) / 2:
                # If the majority have </think>, we finalize:
                #  Evaluate logprob again so we pick the single best
                final_scores, final_gen, avg_latency, num_req_logprobs = eval_logprob_vllm(
                    prompt_batch,
                    big_model_port=big_model_port,
                    big_model=big_model,
                    requests=requests,
                    temperature=0.0,
                    max_tokens=ltok,
                    logprobs=1
                )
                final_scored = list(zip(prompt_batch, final_scores, final_gen))
                final_scored.sort(key=lambda x: x[1], reverse=True)
                # Keep the single best and 'add' the new big model generation to avoid waste.
                prompt_batch = [final_scored[0][0] + final_scored[0][2]]
                break
            else:
                # double up the prompt_batch for next loop (duplicate each)
                prompt_batch = [x for x in prompt_batch for _ in range(2)]

    # There is a good chance that END OF COT has already been reached
    # Because the highest likelihood will be the one which has concluded (?)
    final_answer = prompt_batch[0]  + "\n So, the answer would be: "
    tokens_used = len(final_answer.split())
    remaining_tokens = max_tokens - tokens_used
    if remaining_tokens > 0:
        final_resp, final_latency = generate_text_vllm(
            final_answer,
            port=big_model_port,
            temperature=temperature,
            max_tokens=remaining_tokens,
            model=big_model
        )
        final_text = final_resp["choices"][0]["text"]
        final_answer += final_text
    else:
        final_text = ""
        if test_logging:
            with open(f"{subfolder_path}/final_text.txt", "w") as f:
                f.write(f"NO FINAL TEXT due to max_tokens: {max_tokens} - tokens_used: {tokens_used} = {remaining_tokens}")

    if test_logging:
        with open(f"{subfolder_path}/final_text.txt", "w") as f:
            f.write(final_text)

        with open(f"{subfolder_path}/full_trace.txt", "w") as f:
            f.write(final_answer)

    return final_answer, usage_data
