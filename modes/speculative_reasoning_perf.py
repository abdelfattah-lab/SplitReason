# modes/spec_rewrite_perf.py
"""
Ultra-pipelined speculative-reasoning driver (bug-fixed).

Only change vs previous version:
* The async token streamer _stream_big() no longer tries to `return`.
  Instead the caller deduces finish_reason from how many tokens arrived.
"""

from __future__ import annotations

import asyncio, datetime as _dt, json, os, time, traceback, uuid
from typing import Any, Dict, List, Tuple
import json
import httpx

# ── constants ────────────────────────────────────────────────────────
SMALL_CHUNK      = 64
STREAM_BUCKET    = 8
NUM_PROBE_TOKENS = 6
MAX_TOTAL_TOKENS = 512
# MAX_TOTAL_TOKENS = 16384
MAX_BIG_SEGMENT  = 128
BIG_CHUNK_CAP    = 32

BIG_OPEN  = "<bigmodel>"
BIG_CLOSE = "</bigmodel>"

# ── helpers ──────────────────────────────────────────────────────────
async def _async_call(fn, *args, **kw):
    return await asyncio.to_thread(fn, *args, **kw)

import traceback
import sys

async def _safe_gen_small(batched_generate_text_vllm, *args, **kw):
    """
    Call batched_generate_text_vllm inside a worker-thread.
    If it fails, dump the entire traceback (from that thread)
    to stderr, then re-raise so the outer coroutine stops.
    """
    try:
        return await _async_call(batched_generate_text_vllm, *args, **kw)
    except Exception as e:
        # format_exc() captures the traceback of the exception in this context
        tb = traceback.format_exc()
        print("\n=== small model call failed ===", file=sys.stderr)
        print(tb, file=sys.stderr, flush=True)
        raise RuntimeError(f"small-model call failed: {e}") from e


async def _prefill_other(prompt, port, model, temperature, req_lib, gen_fn):
    return {"choices": [{"text": ""}]}
    # await _async_call(
    #     gen_fn,
    #     prompts=[prompt],
    #     port=port,
    #     temperature=temperature,
    #     max_tokens=1,
    #     model=model,
    #     requests=req_lib,
    # )


async def _stream_big(prompt: str, *, port: int, model: str,
                      temperature: float, client: httpx.AsyncClient):
    """Async generator that yields one token at a time."""
    url = f"http://localhost:{port}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": BIG_CHUNK_CAP,
        "temperature": temperature,
        "stream": True,
        "n": 1,
    }
    async with client.stream("POST", url, json=payload, timeout=None) as r:
        async for line in r.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                line = line[6:]
            if line.strip() in ("[DONE]", "data: [DONE]"):
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                choice = obj["choices"][0]
            except:
                import pdb; pdb.set_trace()
            yield (choice.get("delta", {}).get("content") or
                   choice.get("text", ""))

# ■■■ main coroutine ■■■ ------------------------------------------------
async def _run_speculative_async(
    question, sgen, stok, sdecay, ltok, max_tokens, temperature,
    big_model, big_port, small_model, small_port,
    requests, gen_small, _, __, term_str,
    test_logging, lbound, max_it, seq_scale, tok_counter
) -> Tuple[str, List[Dict[str, Any]]]:

    def _clean(t):  # strip special markers
        for s in ("<｜User｜>", "<｜Assistant｜>", "<｜begin▁of▁sentence｜>",
                  "<｜end▁of▁sentence｜>", "<think>"):
            t = t.replace(s, "")
        return t

    big_hint = "You always use <bigmodel>...</bigmodel> to mark parts of the reasoning process that are important."
    cur = (f"<｜begin▁of▁sentence｜><｜User｜>{_clean(question)}\n"
           f"{big_hint}\n{term_str}<｜Assistant｜>\n<think>")

    usage: List[Dict[str, Any]] = []
    start = time.time()
    stage, big_seg = "small", 0
    done = False
    prefill_big = prefill_small = None

    async with httpx.AsyncClient(timeout=None) as client:
        while not done:
            # ――― small model ―――
            if stage == "small":
                if prefill_big is None or prefill_big.done():
                    prefill_big = asyncio.create_task(
                        _prefill_other(cur, big_port, big_model,
                                       temperature, requests, gen_small)
                    )
                # resp, _ = await _async_call(
                #     gen_small, prompts=[cur], port=small_port,
                #     temperature=temperature,
                #     max_tokens=min(SMALL_CHUNK,
                #                    MAX_TOTAL_TOKENS - tok_counter(cur)),
                #     model=small_model, is_bigmodel_halting=True,
                #     requests=requests,
                # )
                # if resp is None:
                #     raise RuntimeError("small-model call failed")
                remain = MAX_TOTAL_TOKENS - tok_counter(cur)
                if remain <= 0:
                    done = True
                    break
                resp, _ = await _safe_gen_small(
                    batched_generate_text_vllm=gen_small,
                    prompts=[cur],
                    port=small_port,
                    temperature=temperature,
                    max_tokens=max(1, min(SMALL_CHUNK, remain)),
                    model=small_model,
                    is_bigmodel_halting=True,
                    requests=requests,
                )
                delta = resp[0]["choices"][0]["text"]
                fin   = resp[0]["choices"][0]["finish_reason"]
                cur  += delta

                if BIG_OPEN in delta:
                    stage, big_seg = "big", 0
                    continue
                if fin == "stop":
                    done = True
                continue

            # ――― big model (stream) ―――
            if stage == "big":
                if big_seg >= MAX_BIG_SEGMENT:
                    cur += BIG_CLOSE
                    stage = "small"
                    continue

                if prefill_small is None or prefill_small.done():
                    prefill_small = asyncio.create_task(
                        _prefill_other(cur, small_port, small_model,
                                       temperature, requests, gen_small)
                    )

                bucket: List[str] = []
                tokens_this_stream = 0
                async for tok in _stream_big(
                    prompt=cur.replace(big_hint, "")
                              .replace(BIG_CLOSE, "")
                              .replace(BIG_OPEN, ""),
                    port=big_port, model=big_model,
                    temperature=temperature, client=client,
                ):
                    bucket.append(tok)
                    tokens_this_stream += 1
                    big_seg += 1
                    cur += tok

                    if len(bucket) >= STREAM_BUCKET:
                        # import pdb; pdb.set_trace()
                        prefixes = [cur[:-len("".join(bucket))] +
                                    "".join(bucket[:i])
                                    for i in range(1, len(bucket)+1)]
                        probe, _ = await _async_call(
                            gen_small, prompts=prefixes, port=small_port,
                            temperature=temperature, max_tokens=NUM_PROBE_TOKENS,
                            model=small_model, requests=requests,
                        )
                        handoff = None
                        for idx in range(len(prefixes)-1, -1, -1):
                            if probe[idx]["choices"][0]["text"].lstrip(
                            ).startswith(BIG_CLOSE[:4]):
                                handoff = idx
                                break
                        if handoff is not None:
                            chosen = prefixes[handoff]
                            pre = probe[handoff]["choices"][0]["text"
                                   ].split(BIG_CLOSE, 1)[0]
                            cur = chosen + pre + BIG_CLOSE
                            stage = "small"
                            break
                        bucket.clear()

                # stream ended naturally
                if stage == "small":
                    continue      # we broke for hand-off, loop again

                # if server streamed exactly BIG_CHUNK_CAP we assume 'length'
                if tokens_this_stream == BIG_CHUNK_CAP:
                    continue      # ask big model for next chunk
                done = True       # else ('stop') answer finished

    # ――― bookkeeping ―――
    tot = time.time() - start
    ntok = tok_counter(cur)
    csv = "speculative_reasoning_benchmarks_curr.csv"
    if not os.path.exists(csv):
        with open(csv, "w") as f:
            f.write("uuid,small_model,big_model,total_tokens,total_time,time_per_tok\n")
    with open(csv, "a") as f:
        f.write(f"{uuid.uuid4()},{small_model},{big_model},"
                f"{ntok},{tot:.4f},{tot/max(1,ntok):.6f}\n")
    return cur, usage

# ── public blocking wrapper ──────────────────────────────────────────
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
    try:
        return asyncio.run(
            _run_speculative_async(
                question              = question,
                sgen                  = sgen,
                stok                  = stok,
                sdecay                = sdecay,
                ltok                  = ltok,
                max_tokens            = max_tokens,
                temperature           = temperature,
                big_model             = big_model,
                big_port              = big_model_port,
                small_model           = small_model,
                small_port            = small_model_port,
                requests              = requests,
                gen_small             = batched_generate_text_vllm,
                _                     = batched_eval_logprob_vllm,          # unused inside
                __                    = batched_generate_text_with_tokens_vllm,  # unused inside
                term_str              = terminating_string,
                test_logging          = test_logging,
                lbound                = lbound,
                max_it                = max_iterations,
                seq_scale             = sequential_scale,
                tok_counter           = token_counter,
            )
        )
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return f"[ERROR] {e}", []
