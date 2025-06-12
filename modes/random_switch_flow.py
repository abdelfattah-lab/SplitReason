import asyncio
import datetime as _dt
import os
import random
import time
import traceback
import uuid
from typing import Any, Dict, List, Tuple

###############################################################################
# Random‑Switch Flow                                                         #
# ------------------------------------------------------------------------- #
# This implementation keeps the public API identical to the original        #
# `run_random_switch_flow` function while modernising its internals.        #
#                                                                           #
#  *   Asynchronous – the heavy network call to `batched_generate_text_vllm`
#      is executed in a worker thread so that the event‑loop stays live.    #
#  *   Robust – every remote call is wrapped in a protective layer that     #
#      prints the full traceback before re‑raising, preventing silent fail. #
#  *   Random – the decision to use the big or small model is made purely   #
#      by RNG according to `switch_ratio`; no speculative hand‑off logic.   #
###############################################################################

def _strip_control_tokens(text: str) -> str:
    """Remove special control tokens that sometimes appear in prompts."""
    for t in ("<｜User｜>", "<｜Assistant｜>", "<｜begin▁of▁sentence｜>",
              "<｜end▁of▁sentence｜>", "<think>"):
        text = text.replace(t, "")
    return text

async def _async_call(fn, *args, **kwargs):
    """Run *blocking* `fn` in a worker‑thread and return its result."""
    import asyncio
    return await asyncio.to_thread(fn, *args, **kwargs)

async def _safe_generate(
    batched_generate_text_vllm,
    prompt: str,
    *,
    port: int,
    model: str,
    temperature: float,
    max_tokens: int,
    requests,
):
    """Thin async wrapper around `batched_generate_text_vllm` with traceback."""
    try:
        return await _async_call(
            batched_generate_text_vllm,
            prompts=[prompt],
            port=port,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
            requests=requests,
        )
    except Exception as e:
        tb = traceback.format_exc()
        print("\n=== generation failed ===", file=sys.stderr)
        print(tb, file=sys.stderr, flush=True)
        raise RuntimeError(f"generation failed: {e}") from e


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
    batched_generate_text_vllm,  # callable
    batched_eval_logprob_vllm,   # kept for signature compatibility
    switch_ratio: int,
    switch_chunk: int,
    requests,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Modern random switcher that alternates big/small models stochastically.

    The probability of selecting *big_model* each step is ``1/(switch_ratio+1)``
    (e.g. ``switch_ratio=1``  →  50‑50).  At every step we request exactly
    ``switch_chunk`` tokens from the chosen model until we either encounter a
    ``finish_reason == 'stop'`` or reach ``max_tokens``.
    """

    async def _run_async() -> Tuple[str, List[Dict[str, Any]]]:
        usage: List[Dict[str, Any]] = []
        rng = random.Random()
        p_big = 1.0 / (switch_ratio + 1.0)
        start = time.time()


        def _clean(t):  # strip special markers
            for s in ("<｜User｜>", "<｜Assistant｜>", "<｜begin▁of▁sentence｜>",
                    "<｜end▁of▁sentence｜>", "<think>"):
                t = t.replace(s, "")
            return t

        big_hint = ""
        term_str = "\n Put your final answer within \\boxed{}."
        cur = (f"<｜begin▁of▁sentence｜><｜User｜>{_clean(question)}\n"
            f"{big_hint}{term_str}<｜Assistant｜>\n<think>\n")
        current_text   = cur
        tokens_emitted = 0
        big_tokens     = 0
        big_calls      = 0

        step = 0
        while tokens_emitted < max_tokens:
            # ───── decide which model to use this round ─────
            use_big = rng.random() < p_big
            model   = big_model  if use_big else small_model
            port    = big_model_port if use_big else small_model_port
            if use_big:
                big_calls   += 1
                big_tokens  += switch_chunk

            # ───── call remote model ─────
            resp, _ = await _safe_generate(
                batched_generate_text_vllm,
                current_text,
                port=port,
                model=model,
                temperature=temperature,
                max_tokens=switch_chunk,
                requests=requests,
            )

            delta        = resp[0]["choices"][0]["text"]
            finish_reason = resp[0]["choices"][0]["finish_reason"] or "length"
            step        += 1

            current_text   += delta
            tokens_emitted += switch_chunk

            if finish_reason == "stop":
                break

        tot = time.time() - start

        try:
            csv = "random_switcher.csv"
            now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new = not os.path.exists(csv)
            with open(csv, "a", encoding="utf‑8") as fh:
                if new:
                    fh.write("uuid,big_calls,big_tokens,total_tokens,total_time,time_per_tok,p_big,datetime\n")
                fh.write(
                    f"{uuid.uuid4()},{big_calls},{big_tokens},{tokens_emitted},{tot},"
                    f"{tot / tokens_emitted if tokens_emitted > 0 else 0},"
                    f"{p_big},{now}\n"
                )
        except Exception:
            pass

        return current_text, usage

    return asyncio.run(_run_async())
