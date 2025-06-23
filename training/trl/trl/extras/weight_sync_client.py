# extras/weight_sync_client.py
import io, json, time, typing, requests
import torch
from transformers import PreTrainedTokenizerBase
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np         #  add import at top
from tqdm import tqdm
from typing import Optional, List
import typing
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

class WeightSyncClient:
    """
    Client that talks to *spec_service.py*.

    End‑points used
    ---------------
    POST /update_param   – hot‑swap one tensor
    POST /reset_cache    – clear KV‑cache so fresh weights are used
    POST /speculative_reason  – generate (spec_reason_perf=True)
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        spec_endpoint: str = "http://localhost:5002",                 # e.g. "http://localhost:5002"
        timeout: float = 1200.0,
    ):
        self.tok      = tokenizer
        self.base_url = spec_endpoint.rstrip("/")
        self.timeout  = timeout

    # ------------------------------------------------------------
    # 1) Push a single parameter into the live small‑model worker
    # # ------------------------------------------------------------
    # def update_named_param(self, name: str, tensor: torch.Tensor) -> None:
    #     """
    #     Send `tensor` (CUDA/CPU) to /update_param.

    #     spec_service → WorkerExtension.update_weight() → load_weights()
    #     """
    #     t_cpu  = tensor.detach().cpu().contiguous()
    #     buf    = io.BytesIO(t_cpu.numpy().tobytes())
    #     data   = {
    #         "name":  name,
    #         "dtype": str(t_cpu.numpy().dtype),
    #         "shape": json.dumps(list(t_cpu.shape)),
    #     }
    #     files  = {"blob": ("tensor.bin", buf.getvalue())}
    #     r = requests.post(f"{self.base_url}/update_param",
    #                       data=data, files=files, timeout=self.timeout)
    #     r.raise_for_status()


    # def update_named_param(self, name: str, tensor: torch.Tensor) -> None:
    #     """
    #     Send `tensor` to /update_param, BF16‑safe.
    #     """
    #     t_cpu = tensor.detach().cpu().contiguous()

    #     # ---------- BF16 SAFE SERIALISATION ----------
    #     if t_cpu.dtype == torch.bfloat16:
    #         arr = t_cpu.view(torch.uint16).numpy()   # raw 16-bit payload
    #         dtype_str = "bfloat16"
    #     else:
    #         arr = t_cpu.numpy()
    #         dtype_str = str(arr.dtype)
    #     # ---------------------------------------------

    #     buf = io.BytesIO(arr.tobytes())
    #     data = {
    #         "name":  name,
    #         "dtype": dtype_str,                   # string, parsed server‑side
    #         "shape": json.dumps(list(t_cpu.shape)),
    #     }
    #     files = {"blob": ("tensor.bin", buf.getvalue())}

    #     r = requests.post(f"{self.base_url}/update_param",
    #                     data=data, files=files, timeout=self.timeout)
    def update_named_param(self, name: str, tensor: torch.Tensor) -> None:
        t_cpu = tensor.detach().cpu().contiguous()

        if t_cpu.dtype == torch.bfloat16:
            arr        = t_cpu.view(torch.uint16).numpy()   # raw bytes
            dtype_str  = "uint16_bfloat"                    # <- new label
        else:
            arr        = t_cpu.numpy()
            dtype_str  = str(arr.dtype)

        buf  = io.BytesIO(arr.tobytes())
        data = {"name": name,
                "dtype": dtype_str,
                "shape": json.dumps(list(t_cpu.shape))}
        files = {"blob": ("tensor.bin", buf.getvalue())}

        requests.post(f"{self.base_url}/update_param",
                    data=data, files=files, timeout=self.timeout).raise_for_status()
        # ------------------------------------------------------------
    # 2) Flush KV‑cache on the server (call after *all* params done)
    # ------------------------------------------------------------
    def reset_prefix_cache(self) -> None:
        requests.post(f"{self.base_url}/reset_cache", timeout=self.timeout).raise_for_status()

    # # ------------------------------------------------------------
    # # 3) Text generation via /speculative_reason (perf mode)
    # # ------------------------------------------------------------
    # def generate(
    #     self,
    #     prompts: list[str],
    #     n: int,
    #     repetition_penalty: float,
    #     temperature: float,
    #     top_p: float,
    #     top_k: int,
    #     min_p: float,
    #     max_tokens: int,
    #     guided_decoding_regex: typing.Optional[str] = None,   # kept for API compatibility
    # ) -> list[list[int]]:
    #     """
    #     Returns *token‑ids* (NOT strings) – exactly what GRPOTrainer expects.

    #     Shape:  len(prompts) * n  (same order as input: [p0_0, p0_1, …, p1_0, …])
    #     """
    #     completions: list[list[int]] = []

    #     payload_template = {
    #         # fixed knobs that come from GRPOTrainer’s sampling args
    #         "spec_reason_perf": True,
    #     }

    #     for prompt in prompts:
    #         for _ in range(n):
    #             payload = {**payload_template, "question": prompt}
    #             r = requests.post(f"{self.base_url}/speculative_reason",
    #                               json=payload, timeout=self.timeout)
    #             r.raise_for_status()
    #             text = r.json()["final_answer"]
    #             # print(text)
    #             ids  = self.tok.encode(text, add_special_tokens=False)
    #             completions.append(ids)

    #     return completions


    # def generate(
    #     self,
    #     prompts: List[str],
    #     n: int,
    #     repetition_penalty: float,
    #     temperature: float,
    #     top_p: float,
    #     top_k: int,
    #     min_p: float,
    #     max_tokens: int,
    #     guided_decoding_regex: Optional[str] = None,
    # ) -> List[List[int]]:
    #     """
    #     Returns a flat list of token-ID sequences:
    #     [p0_sample0, p0_sample1, …, p1_sample0, …]
    #     Added: tqdm progress bar + per-item status line.
    #     """
    #     payload_template = {"spec_reason_perf": True}

    #     # ❶ Pre-allocate result container
    #     total_reqs = len(prompts) * n
    #     completions: List[List[int]] = [None] * total_reqs

    #     # ❷ Re-use one session
    #     sess = requests.Session()
    #     sess.headers.update({"Connection": "keep-alive"})

    #     def _one_call(prompt: str, slot: int):
    #         payload = {**payload_template, "question": prompt}
    #         r = sess.post(
    #             f"{self.base_url}/speculative_reason",
    #             json=payload,
    #             timeout=self.timeout,
    #         )
    #         r.raise_for_status()
    #         text = r.json()["final_answer"]
    #         completions[slot] = self.tok.encode(text, add_special_tokens=False)

    #     # ❸ Launch in parallel with live progress feedback
    #     with ThreadPoolExecutor(max_workers=1) as pool, tqdm(
    #         total=total_reqs,
    #         desc="Generating",
    #         unit="sample",
    #         dynamic_ncols=True,
    #         leave=True,       # keep the bar after completion
    #         bar_format=(
    #             "{l_bar}{bar}| {n_fmt}/{total_fmt} • {elapsed} ↔ {remaining} "
    #             "• ETA {remaining}  "
    #         ),
    #     ) as pbar:
    #         futures = []
    #         for i, prompt in enumerate(prompts):
    #             for j in range(n):
    #                 slot = i * n + j
    #                 futures.append(pool.submit(_one_call, prompt, slot))

    #         for idx, fut in enumerate(as_completed(futures), start=1):
    #             fut.result()          # raises if the request failed
    #             pbar.update(1)
    #             print(f"Item {idx} processed out of {total_reqs} items", flush=True)
    #     return completions

    def generate(
        self,
        prompts: list[str],
        n: int,
        repetition_penalty: float,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        max_tokens: int,
        guided_decoding_regex: typing.Optional[str] = None,
        *,
        server_parallelism: int = 4,   # set to the number of requests your server can run concurrently
    ) -> list[list[int]]:
        """
        Return a flat list of token-ID sequences in the order
            [p0_sample0, p0_sample1, …, p1_sample0, …]
        """

        payload_template = {"spec_reason_perf": True}

        total_reqs   = len(prompts) * n
        completions  = [None] * total_reqs

        def _one_call(prompt: str, slot: int) -> None:
            """Send one HTTP request and fill completions[slot]."""
            with requests.Session() as sess:          # session per thread
                print(f"Posting item {slot} out of {total_reqs}", flush=True)
                start_time = time.time()
                sess.headers.update({"Connection": "keep-alive"})
                payload = {**payload_template, "question": prompt}
                r = sess.post(f"{self.base_url}/speculative_reason",
                              json=payload, timeout=self.timeout)
                r.raise_for_status()
                text = r.json()["final_answer"]
                completions[slot] = self.tok.encode(text, add_special_tokens=False)
                elapsed = time.time() - start_time
                print(f"Processing item {slot} out of {total_reqs} "
                      f"took {elapsed:.2f} seconds", flush=True)
                print(f"[rank-0] finished slot {slot}/{total_reqs-1}", flush=True)

        workers = min(server_parallelism, total_reqs)  # never oversubscribe
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for i, prompt in enumerate(prompts):
                for j in range(n):
                    slot = i * n + j
                    futures.append(pool.submit(_one_call, prompt, slot))

            for fut in as_completed(futures):
                fut.result()  # will raise immediately on first error

        return completions
    # def generate(
    #     self,
    #     prompts: list[str],
    #     n: int,
    #     repetition_penalty: float,
    #     temperature: float,
    #     top_p: float,
    #     top_k: int,
    #     min_p: float,
    #     max_tokens: int,
    #     guided_decoding_regex: typing.Optional[str] = None,
    # ) -> list[list[int]]:
    #     """
    #     Returns a flat list of token‑ID sequences:
    #     [p0_sample0, p0_sample1, …, p1_sample0, …]
    #     """
    #     # payload_template = {
    #     #     "spec_reason_perf": True,
    #     #     "repetition_penalty": repetition_penalty,
    #     #     "temperature": temperature,
    #     #     "top_p": top_p,
    #     #     "top_k": top_k,
    #     #     "min_p": min_p,
    #     #     "max_tokens": max_tokens,
    #     # }
    #     payload_template = {"spec_reason_perf": True}

    #     # ❶ Pre‑allocate result container
    #     total_reqs = len(prompts) * n
    #     completions: list[list[int]] = [None] * total_reqs

    #     # ❷ Use a single Session for connection re‑use
    #     sess = requests.Session()
    #     sess.headers.update({"Connection": "keep-alive"})

    #     def _one_call(prompt: str, slot: int):
    #         payload = {**payload_template, "question": prompt}
    #         print(f"Posting item {slot} out of {total_reqs}", flush=True)
    #         r = sess.post(f"{self.base_url}/speculative_reason",
    #                       json=payload,
    #                       timeout=self.timeout)
    #         r.raise_for_status()
    #         text = r.json()["final_answer"]
    #         print(f"Processing item {slot} out of {total_reqs}", flush=True)
    #         completions[slot] = self.tok.encode(text, add_special_tokens=False)
    #         print(f"[rank-0] finished slot {slot}/{total_reqs-1}", flush=True)

    #     # ❸ Launch in parallel
    #     with ThreadPoolExecutor(max_workers=8) as pool:
    #         futures = []
    #         for i, prompt in enumerate(prompts):
    #             for j in range(n):
    #                 slot = i * n + j            # preserve required order
    #                 futures.append(pool.submit(_one_call, prompt, slot))

    #         # Will raise on the first failed request
    #         for fut in as_completed(futures):
    #             # print(f"Processing item {fut} out of {total_reqs}", flush=True)
    #             fut.result()
    #             # print(f"Item {fut} processed out of {total_reqs} items", flush=True)

    #     return completions