# extras/weight_sync_client.py
import io, json, time, typing, requests
import torch
from transformers import PreTrainedTokenizerBase

import numpy as np         #  add import at top

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
        timeout: float = 600.0,
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

    # ------------------------------------------------------------
    # 3) Text generation via /speculative_reason (perf mode)
    # ------------------------------------------------------------
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
        guided_decoding_regex: typing.Optional[str] = None,   # kept for API compatibility
    ) -> list[list[int]]:
        """
        Returns *token‑ids* (NOT strings) – exactly what GRPOTrainer expects.

        Shape:  len(prompts) * n  (same order as input: [p0_0, p0_1, …, p1_0, …])
        """
        completions: list[list[int]] = []

        payload_template = {
            # fixed knobs that come from GRPOTrainer’s sampling args
            "spec_reason_perf": True,
        }

        for prompt in prompts:
            for _ in range(n):
                payload = {**payload_template, "question": prompt}
                r = requests.post(f"{self.base_url}/speculative_reason",
                                  json=payload, timeout=self.timeout)
                r.raise_for_status()
                text = r.json()["final_answer"]
                # print(text)
                ids  = self.tok.encode(text, add_special_tokens=False)
                completions.append(ids)

        return completions
