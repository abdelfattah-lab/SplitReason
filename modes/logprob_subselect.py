# modes/logprob_subselect.py

from typing import List, Tuple

def eval_logprob_vllm(
    text_batch: List[str],
    big_model_port: int,
    big_model: str,
    requests,
    temperature: float = 0.0,
    max_tokens: int = 0,
    logprobs: int = 1
) -> List[float]:
    """
    Evaluate the total log-likelihood of each string in text_batch using
    the big model. Returns a list of total log-likelihood scores (higher is better).

    Under the hood, we do one request per string to vLLM with max_tokens=0,
    and 'prompt_logprobs' or 'logprobs' set so that the model returns the
    log probabilities of the prompt tokens. Then we sum them up.
    
    NOTE: Some versions of vLLM accept a list of prompts in a single request.
    For simplicity here, we do one request per string. You can optimize if desired.
    """
    scores = []
    url = f"http://localhost:{big_model_port}/v1/completions"

    for text in text_batch:
        payload = {
            "model": big_model,
            "prompt": text,
            # We generate 0 tokens because we only want to evaluate logprobs
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Use the logprobs feature to get log-likelihood.
            # If you want to see logprobs for each prompt token, use 'prompt_logprobs'.
            # If you want them for the *generated* tokens (which are zero here),
            # you'd use `logprobs`. But here we demonstrate prompt_logprobs.
            "prompt_logprobs": logprobs
        }

        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        resp_json = resp.json()

        # The vLLM response typically includes something like:
        #   {
        #     "choices": [
        #       {
        #         "text": "",
        #         "prompt_logprobs": {
        #             "tokens": [...],
        #             "token_logprobs": [...],
        #             "top_logprobs": [...],
        #             ...
        #         }
        #       }
        #     ]
        #   }
        #
        # We need to parse that to get the total log probability.
        choice = resp_json["choices"][0]
        prompt_loginfo = choice.get("prompt_logprobs", None)
        if not prompt_loginfo:
            # If we didn't get logprobs for some reason, fallback score
            scores.append(float("-inf"))
            continue

        token_logprobs = prompt_loginfo.get("token_logprobs", [])
        # sum them up
        total_ll = sum(token_logprobs)
        scores.append(total_ll)

    return scores


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
):
    """
    Skeleton logic for "logprob subselect" approach. The idea:
      1) We have a batch size = sgen (e.g. 8).
      2) The small model generates 'stok' tokens in parallel for each of sgen candidates.
      3) We send all sgen new candidates to the large model for a log-likelihood eval.
      4) Keep top (sgen / sdecay) (by total log-likelihood).
      5) Optionally let the large model generate ltok tokens on those filtered candidates,
         or let the small model do more tokens, etc.
      6) Repeat until the batch size is 1 or we reach a stopping condition.
      7) Return the final single candidate. Possibly do a final generation with the big model.

    This is only a basic skeleton. You would adapt to your own usage.
    """

    # Keep track of usage data if you like:
    usage_data = []

    # Start with a single shared prompt (the user question):
    base_prompt = f"<|user|>\n{question}\n<|assistant|>\n"

    # For the first step, we replicate that prompt sgen times.
    # Each item in prompt_batch is a partial text that we expand.
    prompt_batch = [base_prompt for _ in range(sgen)]

    # We'll do a while-loop until the batch shrinks to 1 candidate
    # or until some iteration limit.
    iteration_count = 0
    while len(prompt_batch) > 1:
        iteration_count += 1

        # 1) The small model generates `stok` tokens in parallel for each prompt
        #    We'll do it one by one or in parallel. vLLM supports multiple prompts if you pass prompt=[...].
        #    But here is a simple version that does them individually:
        new_candidates = []
        for prompt_text in prompt_batch:
            # Generate stok tokens from the small model
            # We can pass n=1, top_k, etc. Or we can rely on temperature-based sampling
            # so each item is slightly different. But for truly distinct parallel expansions,
            # you'd want n=1 for each item, but you do it sgen times. The code below is simplistic:
            partial_resp, latency = generate_text_vllm(
                prompt_text,
                port=small_model_port,
                temperature=temperature,
                max_tokens=stok,
                model=small_model,
            )
            # If you want usage data, parse partial_resp["usage"] here
            raw_reply = partial_resp["choices"][0]["text"]
            # Build the new prompt by appending the small-model's continuation
            new_candidates.append(prompt_text + raw_reply)

        # Now we have 'new_candidates' of the same size as prompt_batch
        # But in many “parallel generation” approaches, you'd actually do
        # a single request with prompt=[prompt_batch], n=1, etc.
        # This code is deliberately simplistic.

        # 2) Evaluate each candidate with the big model to get a log-likelihood
        #    via eval_logprob_vllm:
        scores = eval_logprob_vllm(
            new_candidates,
            big_model_port=big_model_port,
            big_model=big_model,
            requests=requests,
            temperature=0.0,    # Typically 0.0 for pure log-likelihood
            max_tokens=0,
            logprobs=1
        )

        # 3) Keep the top (sgen / sdecay) candidates
        n_keep = max(1, len(scores) // sdecay)
        # Pair up each candidate with its logprob
        scored_candidates = list(zip(new_candidates, scores))
        # Sort descending by total log-likelihood
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        # Slice top n_keep
        prompt_batch = [x[0] for x in scored_candidates[:n_keep]]

        # 4) Optionally, let the large model generate `ltok` tokens for each of the top ones:
        if ltok > 0:
            refined_candidates = []
            for text in prompt_batch:
                part_resp, part_latency = generate_text_vllm(
                    text,
                    port=big_model_port,
                    temperature=temperature,
                    max_tokens=ltok,
                    model=big_model
                )
                # usage if needed
                extra_reply = part_resp["choices"][0]["text"]
                refined_candidates.append(text + extra_reply)
            prompt_batch = refined_candidates

        # 5) Now we go back to step #1, letting the small model add `stok` more tokens again,
        #    then big model logprob, etc., until prompt_batch length is 1 or we hit some iteration limit.
        if len(prompt_batch) == 1:
            # We are done
            break

        # You might also want a maximum iteration limit, or a check on max tokens, etc.
        if iteration_count > 10:
            # Arbitrary safety
            break

    # If we exit the loop with a single final candidate, that is our best text.
    final_answer = prompt_batch[0]

    # If you want, do a final completion with the big model to get a high-quality tail:
    #  (But only if you haven't already done it above)
    final_resp, final_latency = generate_text_vllm(
        final_answer,
        port=big_model_port,
        temperature=temperature,
        max_tokens=(max_tokens - len(final_answer.split())),
        model=big_model
    )
    final_text = final_resp["choices"][0]["text"]
    final_answer = final_answer + final_text

    return final_answer, usage_data
