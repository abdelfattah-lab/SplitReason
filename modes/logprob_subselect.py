import traceback
from typing import List, Tuple
import random
from tqdm import tqdm

# python test_spec.py  --logprob_subselect --stok 256 --sgen 32 --sdecay 2 --ltok 16 --lbound 2

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

    for text in tqdm(text_batch):
        try:
            payload = {
                "model": big_model,
                "prompt": text,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "logprobs": logprobs,
                # "echo": True   # If you need token-level breakdown of the prompt
            }
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
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
            prompt_ll = sum(token_logprobs) if len(token_logprobs) > 0 else 0.0
            scores.append(prompt_ll)
            gentexts.append(generated_text)

        except Exception as e:
            traceback.print_exc()
            print("\n\n[WARNING]: Using Random Log-Prob Assignment, "
                  f"failed to get logprobs for text: {text}\n\n")
            scores.append(random.random())
            gentexts.append("")

    return scores, gentexts


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
        draft_logs = "draft_logs"
        import os
        if not os.path.exists(draft_logs):
            os.makedirs(draft_logs)

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

    while len(prompt_batch) > 1 and iteration_count < max_iterations:
        iteration_count += 1

        # --- Step 1) Small model generation: stok tokens for each prompt
        new_candidates = []
        for prompt_text in prompt_batch:
            # Generate stok tokens from the small model (1 sample per prompt here)
            partial_resp, latency = generate_text_vllm(
                prompt_text,
                port=small_model_port,
                temperature=temperature,
                max_tokens=stok,
                model=small_model,
            )
            raw_reply = partial_resp["choices"][0]["text"]
            new_candidates.append(prompt_text + raw_reply)

        # --- Step 4) Large model expansions (ltok tokens)
        #     (We "always" do this now that ltok>0 is required)
        scores, big_generations = eval_logprob_vllm(
            new_candidates,
            big_model_port=big_model_port,
            big_model=big_model,
            requests=requests,
            temperature=0.0,
            max_tokens=ltok,  # let the big model actually generate tokens
            logprobs=1
        )

        # Nicely print the new candidates, their scores and the big_generations in a for loop for each candidate
        for i, (cand, score, big_gen) in enumerate(zip(new_candidates, scores, big_generations)):
            print(f"--- Candidate {i+1} \t \t Score: {score} ---")
            print(cand)
            print(f"--- BigModel Continuation {i+1} ---")
            print(big_gen)
            print("\n\n")
            if test_logging:
                write_model_name = big_model.split("/")[-1]
                with open(f"{draft_logs}/{write_model_name}_iter{iteration_count}.txt", "a") as f:
                    f.write("\n" + "-" * 80 + "\n" + "Candidate\t \t Score: " + str(score) + "\n" + "-" * 80 + "\n")
                    f.write(cand)
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
                final_scores, final_gen = eval_logprob_vllm(
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
                # Keep the single best
                prompt_batch = [final_scored[0][0] + final_scored[0][2]]
                break
            else:
                # Otherwise, we "double" the variants using the small model
                # We'll generate 2 expansions per candidate, for example.
                doubled_candidates = []
                for prompt_text in prompt_batch:
                    # generate 2 expansions from the small model
                    for _ in range(2):
                        partial_resp, latency = generate_text_vllm(
                            prompt_text,
                            port=small_model_port,
                            temperature=temperature,
                            max_tokens=stok,
                            model=small_model,
                        )
                        raw_reply = partial_resp["choices"][0]["text"]
                        doubled_candidates.append(prompt_text + raw_reply)
                # Evaluate them all and keep the top again:
                d_scores, _ = eval_logprob_vllm(
                    doubled_candidates,
                    big_model_port=big_model_port,
                    big_model=big_model,
                    requests=requests,
                    temperature=0.0,
                    max_tokens=0,
                    logprobs=1
                )
                scored_doubled = list(zip(doubled_candidates, d_scores))
                scored_doubled.sort(key=lambda x: x[1], reverse=True)
                # Maybe we revert to the "original beam size" or keep 2*lbound, etc.
                # For simplicity, let's just keep sgen if you want to push the beam back up:
                n_keep2 = min(len(scored_doubled), sgen)
                prompt_batch = [x[0] for x in scored_doubled[:n_keep2]]

        # If the batch still has more than 1 candidate, we continue the loop
        # Possibly we break if iteration_count hits max_iterations

    # If we exit the loop with a single final candidate, that is our best text.
    final_answer = prompt_batch[0]

    # Optionally, do a final completion with the big model for the rest of the tokens
    # If you want to ensure you haven't exceeded max_tokens, you might measure how many
    # tokens we've consumed so far, etc. Here, we do a naive approach:
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

    return final_answer, usage_data
