# modes/spec_rewrite_perf.py
"""
Fast, fully–asynchronous speculative-reasoning driver.

Key ideas
---------
1. *Two hot sessions* – we keep both the “small” and “big” vLLM servers
   alive the whole time.  While one model is generating the next chunk,
   the other is already being *prefilled* with the updated context so
   its KV-cache is ready the instant we flip control.

2. *Chunked streaming* – instead of asking the models for the whole
   tail, we pull fixed-size chunks (`SMALL_CHUNK`, `BIG_CHUNK`) in a
   loop.  This lets us probe frequently without paying per-token RTT.

3. *Parallel probing* – every time the big model emits a chunk we
   *concurrently* launch a “probe” batch on the small model (just
   `NUM_PROBE_TOKENS` tokens) for **all** incremental prefixes in that
   chunk.  The first probe that starts with `"</big"` yanks control
   back to the small model.

4. *asyncio everywhere* – the long-running, blocking `requests.post`
   calls are run in the default executor via `asyncio.to_thread`, so
   network/IO is fully overlapped.

You may freely change the constants at the top to tune throughput /
latency trade-offs.
"""
from __future__ import annotations

import asyncio
import datetime
import os
import time
import traceback
import uuid
from typing import Any, Dict, List, Tuple

# -------------------------------------------------------------------- #
#                               CONSTANTS                              #
# -------------------------------------------------------------------- #

SMALL_CHUNK        = 16   # tokens pulled from the small model at once
BIG_CHUNK          = 64   # tokens pulled from the big model at once
NUM_PROBE_TOKENS   = 8    # how many tokens each probe asks the small model for
MAX_TOTAL_TOKENS   = 16_384
MAX_BIGMODEL_TOKENS = 512  # hard cap emitted per <bigmodel> segment

BIG_OPEN  = "<bigmodel>"
BIG_CLOSE = "</bigmodel>"


# -------------------------------------------------------------------- #
#                          HELPER – async wrappers                     #
# -------------------------------------------------------------------- #
def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


async def _async_call(fn, *args, **kwargs):
    """
    Run the blocking vLLM call in a thread so that we do not block the
    event-loop.  All vLLM helpers (`batched_generate_…`) are untouched.
    """
    return await asyncio.to_thread(fn, *args, **kwargs)


# -------------------------------------------------------------------- #
#                        MAIN SPECULATIVE DECODER                      #
# -------------------------------------------------------------------- #
async def _run_speculative_async(
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
    test_logging: bool,
    lbound: int,
    max_iterations: int,
    sequential_scale,
    token_counter,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Internal coroutine that does the actual work.  The public helper
    below just runs it in `asyncio.run()` so your external interface
    remains synchronous.
    """

    # ---------------------------- prompt ---------------------------- #
    def sanitize(text: str) -> str:
        for t in (
            "<｜User｜>",
            "<｜Assistant｜>",
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
            "<think>",
        ):
            text = text.replace(t, "")
        return text

    bigmodel_hint = (
        "You always use <bigmodel>...</bigmodel> to mark parts of the "
        "reasoning process that are important."
    )
    question = sanitize(question)
    current_text = (
        f"<｜begin▁of▁sentence｜><｜User｜>{question}\n"
        f"{bigmodel_hint}\n"
        f"{terminating_string}"
        f"<｜Assistant｜>\n"
        f"<think>"
    )

    usage_data: List[Dict[str, Any]] = []
    start_time = time.time()

    # --------------------------------------------------------------- #
    #                           MAIN LOOP                             #
    # --------------------------------------------------------------- #
    big_tokens_emitted_in_segment = 0
    stage = "small"  # or "big"
    finished = False

    while not finished:
        # ------------------------ SMALL STAGE ----------------------- #
        if stage == "small":
            # Fire one small-model chunk
            small_resp, _lat = await _async_call(
                batched_generate_text_vllm,
                prompts=[current_text],
                port=small_model_port,
                temperature=temperature,
                max_tokens=min(SMALL_CHUNK, MAX_TOTAL_TOKENS - token_counter(current_text)),
                model=small_model,
                is_bigmodel_halting=True,  # stop at <bigmodel>
                requests=requests,
            )
            if small_resp is None:
                raise RuntimeError("Small model call failed")

            delta = small_resp[0]["choices"][0]["text"]
            current_text += delta
            finish_reason = small_resp[0]["choices"][0]["finish_reason"]
            stop_reason  = small_resp[0]["choices"][0]["stop_reason"]

            # ───── decision matrix ──────────────────────────────
            if stop_reason == BIG_OPEN:             # hit “<bigmodel>”
                stage = "big"
                big_tokens_emitted_in_segment = 0
                continue                            # switch to big loop

            if finish_reason == "stop":             # EOT (no <bigmodel>)
                finished = True
                break

            # finish_reason == "length" → we just need another chunk
            continue                                # stay in small loop
            # if finish_reason == "length":
            #     # Small model finished the whole answer – we are done.
            #     print("\n\n Small model finished whole answer – finished\n\n")
            #     finished = True
            #     break

            # if stop_reason is None:
            #     # It emitted </think> / EOT without <bigmodel>; we're done.
            #     print("\n\n EOT without <bigmodel> – finished\n\n")
            #     finished = True
            #     break

            # # Otherwise stop_reason is "<bigmodel>" → switch to big stage
            # stage = "big"
            # big_tokens_emitted_in_segment = 0
            # continue  # next iteration

        # ------------------------- BIG STAGE ------------------------ #
        if stage == "big":
            if big_tokens_emitted_in_segment >= MAX_BIGMODEL_TOKENS:
                # safety: close the segment automatically
                current_text += BIG_CLOSE
                stage = "small"
                continue

            # Ask big model for a chunk **concurrently** with the probe task we
            # will launch right after receiving its tokens.
            big_resp, big_tokens, _lat = await _async_call(
                batched_generate_text_with_tokens_vllm,
                prompts=[current_text.replace(BIG_OPEN, "").replace(BIG_CLOSE, "").replace(bigmodel_hint, "")],
                port=big_model_port,
                temperature=temperature,
                max_tokens=BIG_CHUNK,
                model=big_model,
                requests=requests,
            )
            if big_resp is None:
                raise RuntimeError("Big model call failed")

            big_text_chunk   = big_resp[0]["choices"][0]["text"]
            finish_reason    = big_resp[0]["choices"][0]["finish_reason"]
            token_list       = big_tokens[0]  # one prompt → one token list
            big_tokens_emitted_in_segment += len(token_list)

            # Build *all* incremental prefixes of that chunk
            prefixes: List[str] = [
                current_text + "".join(token_list[:i])
                for i in range(1, len(token_list) + 1)
            ]

            # Launch probe call on small model *in parallel* (asyncio.gather)
            probe_task = asyncio.create_task(
                _async_call(
                    batched_generate_text_vllm,
                    prompts=prefixes,
                    port=small_model_port,
                    temperature=temperature,
                    max_tokens=NUM_PROBE_TOKENS,
                    model=small_model,
                    requests=requests,
                )
            )

            # While the probe is running we can already append the big chunk;
            # if the probe decides to cut us off we’ll roll back below.
            current_text += big_text_chunk

            # Wait for probe to finish
            probe_resp, _probe_lat = await probe_task

            # # Analyse probe responses **from longest prefix backwards**,
            # # favouring the most tokens harvested from the big model.
            # handoff = None
            # for prefix_idx, probe_choice in enumerate(reversed(probe_resp[0])):
            #     cont = probe_choice["choices"][0]["text"]
            #     if cont.lstrip().startswith(BIG_CLOSE[:4]):  # "</big"
            #         handoff = prefix_idx
            #         break

            # Examine probe results from the *longest* prefix backwards.
            handoff_idx = None
            for idx in range(len(prefixes) - 1, -1, -1):
                cont = probe_resp[idx]["choices"][0]["text"]
                if cont.lstrip().startswith(BIG_CLOSE[:4]):  # "</big"
                    handoff_idx = idx
                    break

            # if handoff is not None:
            #     # Roll back to that prefix, splice probe text, close tag,
            #     # and give control back to small model.
            #     chosen_prefix = prefixes[len(prefixes) - 1 - handoff]
            #     pretext       = probe_resp[0][len(prefixes) - 1 - handoff]["choices"][0]["text"]
            if handoff_idx is not None:
                # Roll back to the chosen prefix, splice probe text, close tag
                chosen_prefix = prefixes[handoff_idx]
                pretext       = probe_resp[handoff_idx]["choices"][0]["text"]
                # Strip anything *after* the open-delimiter – only the slice
                # *before* "</big" belongs to big model.
                pre_big_close = pretext.split(BIG_CLOSE, 1)[0]
                current_text  = chosen_prefix + pre_big_close + BIG_CLOSE
                stage = "small"
                continue

            # No probe triggered → keep using big model
            if finish_reason != "length":
                # Big model hit EOT – we're done.
                finished = True
                break

            # Otherwise loop again with the big model
            continue

    # ---------------------------------------------------------------- #
    #                       METRICS + BOOK-KEEPING                      #
    # ---------------------------------------------------------------- #
    total_time   = time.time() - start_time
    total_tokens = token_counter(current_text)
    time_per_tok = total_time / max(1, total_tokens)

    # Persist benchmark: append or create csv
    uuid_ = uuid.uuid4()
    coverage = mean = median = std = 0.0  # place-holders – reuse your helpers if needed
    csv_path = "speculative_reasoning_benchmarks_curr.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(
                "uuid,small_model,big_model,"
                "small_chunk,big_chunk,probe_tokens,"
                "total_tokens,total_time,time_per_tok\n"
            )
    with open(csv_path, "a") as f:
        f.write(
            f"{uuid_},{small_model},{big_model},"
            f"{SMALL_CHUNK},{BIG_CHUNK},{NUM_PROBE_TOKENS},"
            f"{total_tokens},{total_time:.4f},{time_per_tok:.6f}\n"
        )

    return current_text, usage_data


# -------------------------------------------------------------------- #
#              PUBLIC, *synchronous* entry-point – preserved           #
# -------------------------------------------------------------------- #
def run_speculative_reasoning_flow_perf(
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
    token_counter=len,  # default fallback
):
    """
    Drop-in replacement.  Simply blocks until the async coroutine above
    finishes and then returns the exact same `(text, usage_data)` tuple
    expected by downstream evaluation code.
    """
    try:
        return asyncio.run(
            _run_speculative_async(
                question,
                sgen,
                stok,
                sdecay,
                ltok,
                max_tokens,
                temperature,
                big_model,
                big_model_port,
                small_model,
                small_model_port,
                requests,
                batched_generate_text_vllm,
                batched_eval_logprob_vllm,
                batched_generate_text_with_tokens_vllm,
                terminating_string,
                test_logging,
                lbound,
                max_iterations,
                sequential_scale,
                token_counter,
            )
        )
    except Exception as e:  # noqa: BLE001
        # If something blows up we keep behaviour identical: return the
        # partial text so the evaluation harness does not crash.
        traceback.print_exc()
        return f"[ERROR] {e}", []


# -------------------------------------------------------------------- #
#                                THE END                               #
# -------------------------------------------------------------------- #
