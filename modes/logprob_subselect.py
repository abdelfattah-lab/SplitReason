import traceback
from typing import List, Tuple
import random
from tqdm import tqdm
import os
import datetime
import time
from typing import List, Tuple, Dict, Any, Optional, Union
# python test_spec.py  --logprob_subselect --stok 256 --sgen 32 --sdecay 2 --ltok 16 --lbound 2

### CAN WE DO SLIDING WINDOW WITH QUESTION ANCHOR TOKENS FOR LOGPROB EVAL ON LARGE MODELS
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
    batched_generate_text_vllm,  # <-- use the batched version
    batched_eval_logprob_vllm,   # <-- use the batched version
    terminating_string: str,
    test_logging: bool = False,
    lbound: int = 2,
    max_iterations: int = 100,
    sequential_scale=0,
):
    """
    A "logprob subselect" approach with batched calls to both small_model and big_model.
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

    if "｜" not in question:
        base_prompt = f"<｜begin▁of▁sentence｜><｜User｜>{question}{terminating_string}<｜Assistant｜>\n{model_think_prefix}"
    else:
        base_prompt = f"{question}{terminating_string}\n{model_think_prefix}"

    # For the first step, replicate that prompt sgen times
    prompt_batch = [base_prompt for _ in range(sgen)]

    iteration_count = 0
    small_model_generations = {}

    while len(prompt_batch) > 1 and iteration_count < max_iterations:
        iteration_count += 1

        # --- 1) Generate expansions from the small model in *batch* ---
        try:
            partial_resps, small_model_avg_latency = batched_generate_text_vllm(
                prompts=prompt_batch,
                port=small_model_port,
                temperature=temperature,
                max_tokens=stok,
                model=small_model,
                requests=requests,
            )
        except Exception as e:
            print(f"ERROR: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print("\n\n RETURNING EARLY OUTPUT \n\n")
            return prompt_batch[0], usage_data

        # --- 2) Build the new candidates by appending each partial response to the corresponding prompt ---
        new_candidates = []
        small_model_generations[iteration_count] = []

        for old_prompt, partial_resp in zip(prompt_batch, partial_resps):
            raw_reply = partial_resp["choices"][0]["text"]
            small_model_generations[iteration_count].append(raw_reply)
            new_candidates.append(old_prompt + raw_reply)

        # --- 3) Evaluate log probs (and get continuation) from the big model, in *batch* ---
        try:
            scores, big_generations, big_model_avg_latency, num_req_logprobs = batched_eval_logprob_vllm(
                text_batch=new_candidates,
                big_model_port=big_model_port,
                big_model=big_model,
                requests=requests,
                temperature=0.0,
                max_tokens=ltok,  # the big model is allowed to generate these tokens
                logprobs=1
            )
        except Exception as e:
            print(f"ERROR: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print("\n\n RETURNING EARLY OUTPUT \n\n")
            return prompt_batch[0], usage_data

        # For logging:
        if test_logging:
            for i, (cand, score, big_gen) in enumerate(zip(new_candidates, scores, big_generations)):
                print(f"--- Candidate {i+1} \t \t Score: {score} ---")
                print(small_model_generations[iteration_count][i])
                print(f"--- BigModel Continuation {i+1} ---")
                print(big_gen)
                print("\n\n")

            # Sort by score just for debug
            sort_by_score = sorted(
                zip(new_candidates, scores, big_generations),
                key=lambda x: x[1],
                reverse=True
            )
            write_model_name = big_model.split("/")[-1]
            with open(f"{subfolder_path}/{write_model_name}_iter{iteration_count}.txt", "w") as f:
                for i, (cand, score, big_gen) in enumerate(sort_by_score):
                    f.write("\n" + "-" * 80 + "\n" + f"Candidate {i+1} | Score: {score}\n" + "-" * 80 + "\n")
                    f.write(f"Small model partial:\n{cand[len(base_prompt):]}\n")
                    f.write("\n" + "-" * 80 + "\n" + "BigModel Continuation\n" + "-" * 80 + "\n")
                    f.write(big_gen)
                    f.write("\n\n")

        # --- 4) Combine the new candidates with the big model's generation ---
        combined_candidates = []
        for i, cand in enumerate(new_candidates):
            combined_candidates.append(cand + big_generations[i])

        # --- 5) Sort by score, keep top N, per sdecay ---
        n_keep = max(1, len(scores) // sdecay)
        scored_candidates = list(zip(combined_candidates, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        prompt_batch = [x[0] for x in scored_candidates[:n_keep]]

        # --- 6) If we are now at or below lbound, check majority for </think> ---
        if len(prompt_batch) <= lbound:
            n_with_think = sum(1 for cand in prompt_batch if model_think_suffix in cand)
            if n_with_think > len(prompt_batch) / 2:
                # If majority contain the closing tag, finalize
                try:
                    final_scores, final_gen, final_avg_latency, num_req_logprobs = batched_eval_logprob_vllm(
                        text_batch=prompt_batch,
                        big_model_port=big_model_port,
                        big_model=big_model,
                        requests=requests,
                        temperature=0.0,
                        max_tokens=ltok,
                        logprobs=1
                    )
                except Exception as e:
                    print(f"ERROR: {e}")
                    print(f"Traceback: {traceback.format_exc()}")
                    print("\n\n RETURNING EARLY OUTPUT \n\n")
                    return prompt_batch[0], usage_data
                final_scored = list(zip(prompt_batch, final_scores, final_gen))
                final_scored.sort(key=lambda x: x[1], reverse=True)
                # Keep the single best
                prompt_batch = [final_scored[0][0] + final_scored[0][2]]
                break
            else:
                # Otherwise, "double" the beam by duplicating each candidate
                prompt_batch = [x for x in prompt_batch for _ in range(2)]

    # Possibly all have ended or we ran out of iterations. 
    # We'll do one final extension if we have leftover tokens:
    final_answer = prompt_batch[0] + r"""\n So, finally the answer in the expected format would be: \bo"""

    # tokens_used = len(final_answer.split())
    remaining_tokens = 512
    # One last generation from the big model
    try:
        final_resps, final_latency = batched_generate_text_vllm(
            prompts=[final_answer],
            port=big_model_port,
            temperature=temperature,
            max_tokens=remaining_tokens,
            model=big_model,
            requests=requests
        )
    except Exception as e:
        print(f"ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("\n\n RETURNING EARLY OUTPUT \n\n")
        return prompt_batch[0], usage_data
                        

    final_text = final_resps[0]["choices"][0]["text"]
    final_answer += final_text

    print("final_text: ", final_text)
    if test_logging:
        with open(f"{subfolder_path}/final_text.txt", "w") as f:
            f.write(final_text)

        with open(f"{subfolder_path}/full_trace.txt", "w") as f:
            f.write(final_answer)

    return final_answer, usage_data
