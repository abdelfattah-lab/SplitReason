"""
Ultra-pipelined speculative-reasoning driver (bug‑fixed + trace logging).

Changes versus previous version:
* Added fine‑grained event tracing.  Every meaningful step during decoding
  is timestamped and later flushed to ``decode_trace/{run_id}.csv`` so we can
  visualize who (big/small model, prefill jobs, etc.) did what and when.
* Re‑used the same ``run_id`` for both benchmark and trace files so results
  correlate trivially.
* Minor refactors to keep the public API intact.
"""

from __future__ import annotations

import asyncio, csv, datetime as _dt, json, os, sys, time, traceback, uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple
import httpx

# ── constants ────────────────────────────────────────────────────────
SMALL_CHUNK      = 64
STREAM_BUCKET    = 8
NUM_PROBE_TOKENS = 6
MAX_TOTAL_TOKENS = 16_384
MAX_BIG_SEGMENT  = 128
BIG_CHUNK_CAP    = 32

BIG_OPEN  = "<bigmodel>"
BIG_CLOSE = "</bigmodel>"

DECODE_TRACE_DIR = "decode_trace"  # ← new — where timestamped events go

# ── helpers ──────────────────────────────────────────────────────────
async def _async_call(fn, *args, **kw):
    return await asyncio.to_thread(fn, *args, **kw)

async def _safe_gen_small(batched_generate_text_vllm, *args, **kw):
    """Wrap small‑model call so we dump its traceback if it blows up."""
    try:
        return await _async_call(batched_generate_text_vllm, *args, **kw)
    except Exception as e:
        print("\n=== small model call failed ===", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        raise RuntimeError(f"small‑model call failed: {e}") from e

async def _prefill_other(prompt, port, model, temperature, req_lib, gen_fn):
    """Stub for the prefill RPC to the *other* model (big⇄small)."""
    return {"choices": [{"text": ""}]}

async def _stream_big(prompt: str, *, port: int, model: str,
                      temperature: float, client: httpx.AsyncClient):
    """Async generator that yields *one* token at a time from the big model."""
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
            choice = obj["choices"][0]
            yield (choice.get("delta", {}).get("content") or
                   choice.get("text", ""))

# ── private util: event logging ──────────────────────────────────────

def _ensure_trace_dir() -> Path:
    """Create ``decode_trace`` directory if it doesn't exist."""
    p = Path(DECODE_TRACE_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p

# ■■■ main coroutine ■■■ ------------------------------------------------
async def _run_speculative_async(
    question, sgen, stok, sdecay, ltok, max_tokens, temperature,
    big_model, big_port, small_model, small_port,
    requests, gen_small, _, __, term_str,
    test_logging, lbound, max_it, seq_scale, tok_counter
) -> Tuple[str, List[Dict[str, Any]]]:

    def _clean(t):  # strip special markers so the prompt is human‑readable
        for s in ("<｜User｜>", "<｜Assistant｜>", "<｜begin▁of▁sentence｜>",
                  "<｜end▁of▁sentence｜>", "<think>"):
            t = t.replace(s, "")
        return t

    # ── bookkeeping objects ──
    run_id = uuid.uuid4()
    trace: List[Tuple[float, str, str, str]] = []   # (elapsed, stage, action, note)
    start_time = time.time()

    def _log(stage: str, action: str, note: str = "") -> None:
        """Append an event to the in‑memory trace."""
        trace.append((time.time() - start_time, stage, action, note))
        if test_logging:  # optional console spam when debugging
            print(f"[TRACE] {trace[-1]}")

    big_hint = "You always use <bigmodel>...</bigmodel> to mark parts of the reasoning process that are important."
    cur = (f"<｜begin▁of▁sentence｜><｜User｜>{_clean(question)}\n"
           f"{big_hint}\n{term_str}<｜Assistant｜>\n<think>")

    usage: List[Dict[str, Any]] = []
    stage, big_seg, done = "small", 0, False
    prefill_big = prefill_small = None

    # ── helper wrappers that *also log* ──
    async def _call_small(*args, **kw):
        _log("small", "call_start", f"rem={MAX_TOTAL_TOKENS - tok_counter(cur)}")
        out = await _safe_gen_small(*args, **kw)
        _log("small", "call_end", "✓")
        return out

    async with httpx.AsyncClient(timeout=None) as client:
        while not done:
            # ――― small model ―――
            if stage == "small":
                if prefill_big is None or prefill_big.done():
                    _log("prefill_big", "spawn")
                    prefill_big = asyncio.create_task(
                        _prefill_other(cur, big_port, big_model,
                                       temperature, requests, gen_small)
                    )
                remain = MAX_TOTAL_TOKENS - tok_counter(cur)
                if remain <= 0:
                    done = True
                    break
                resp, _ = await _call_small(
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
                _log("small", "delta", f"{len(delta)}toks")

                if BIG_OPEN in delta:
                    stage, big_seg = "big", 0
                    _log("state", "enter_big")
                    continue
                if fin == "stop":
                    done = True
                continue

            # ――― big model (stream) ―――
            if stage == "big":
                if big_seg >= MAX_BIG_SEGMENT:
                    cur += BIG_CLOSE
                    stage = "small"
                    _log("big", "max_seg_close")
                    continue

                if prefill_small is None or prefill_small.done():
                    _log("prefill_small", "spawn")
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
                    _log("big", "stream_tok", tok.replace("\n", "\\n")[:30])

                    if len(bucket) >= STREAM_BUCKET:
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
                            if probe[idx]["choices"][0]["text"].lstrip(  # noqa: E501
                            ).startswith(BIG_CLOSE[:4]):
                                handoff = idx
                                break
                        if handoff is not None:
                            chosen = prefixes[handoff]
                            pre = probe[handoff]["choices"][0]["text"].split(BIG_CLOSE, 1)[0]
                            cur = chosen + pre + BIG_CLOSE
                            stage = "small"
                            _log("handoff", "big→small")
                            break
                        bucket.clear()

                # stream ended naturally
                if stage == "small":
                    continue      # we broke for hand‑off, loop again

                # if server streamed exactly BIG_CHUNK_CAP we assume 'length'
                if tokens_this_stream == BIG_CHUNK_CAP:
                    continue      # ask big model for next chunk
                done = True       # else ('stop') answer finished

    # ── flush trace to CSV ────────────────────────────────────────────
    trace_dir = _ensure_trace_dir()
    trace_path = trace_dir / f"{run_id}.csv"
    with open(trace_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "elapsed_sec", "stage", "action", "note"])
        writer.writerows([[run_id, *row] for row in trace])

    _log("trace", "saved", str(trace_path))

    # ── final bookkeeping ────────────────────────────────────────────
    tot = time.time() - start_time
    ntok = tok_counter(cur)
    bench_path = Path("speculative_reasoning_benchmarks_curr.csv")
    if not bench_path.exists():
        bench_path.write_text("uuid,small_model,big_model,total_tokens,total_time,time_per_tok\n")
    with bench_path.open("a") as f:
        f.write(f"{run_id},{small_model},{big_model},"
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
