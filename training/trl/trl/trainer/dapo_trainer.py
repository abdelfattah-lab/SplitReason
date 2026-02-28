# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
DAPOTrainer — inherits from GRPOTrainer, overriding only compute_loss.

DAPO modifications (https://huggingface.co/papers/2503.14476):
1. Token-level global normalization: loss / gather(active_tokens) across all processes
2. Overlong filtering: zero out completion_mask for truncated completions (no EOS)
3. Clip-Higher, no reward scaling, no KL — handled via DAPOConfig defaults
"""

import torch
from typing import Optional

from ..extras.profiling import profiling_decorator
from .grpo_trainer import GRPOTrainer


class DAPOTrainer(GRPOTrainer):
    """
    Trainer for the DAPO algorithm. Subclasses GRPOTrainer and overrides only
    compute_loss to add:
    - Overlong filtering (mask_truncated_completions)
    - Token-level global normalization (loss_type="dapo")
    """

    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The DAPOTrainer does not support returning outputs")

        # If loss_type is not "dapo", fall back to parent GRPOTrainer.compute_loss
        loss_type = getattr(self.args, "loss_type", "dapo")
        if loss_type != "dapo":
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # --- Overlong filtering: zero out mask for truncated completions (no EOS) ---
        mask_truncated = getattr(self.args, "mask_truncated_completions", True)
        if mask_truncated:
            eos_token_id = self.processing_class.eos_token_id
            # A completion is truncated if it contains no EOS token at all
            has_eos = (completion_ids == eos_token_id).any(dim=1)  # (B,)
            # Zero out the entire completion_mask for truncated sequences
            completion_mask = completion_mask * has_eos.unsqueeze(1).int()

        # --- Build effective_mask: preserve small_token_mask logic from SplitReason ---
        small_token_mask = inputs.get("small_token_mask", None)
        if small_token_mask is not None:
            effective_mask = completion_mask * small_token_mask
        else:
            effective_mask = completion_mask

        mask_sum = effective_mask.sum()
        if mask_sum == 0:
            effective_mask = completion_mask
            mask_sum = effective_mask.sum()

        # Compute KL divergence (if beta != 0)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the clipped surrogate loss
        advantages = inputs["advantages"]
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # --- DAPO: token-level global normalization ---
        # Sum masked loss on this process, then divide by the total active tokens
        # gathered across ALL processes.
        local_loss_sum = (per_token_loss * effective_mask).sum()
        global_mask_sum = self.accelerator.gather(mask_sum.unsqueeze(0)).sum()
        # Clamp to avoid division by zero if all completions were masked out globally
        global_mask_sum = global_mask_sum.clamp_min(1)
        loss = local_loss_sum / global_mask_sum

        # Log metrics (same as GRPOTrainer)
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * effective_mask).sum() / mask_sum.clamp_min(1)
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * effective_mask).sum() / mask_sum.clamp_min(1)
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )

        # Log how many completions were masked due to truncation
        if mask_truncated:
            has_eos_gathered = self.accelerator.gather_for_metrics(has_eos.float())
            self._metrics[mode]["frac_non_truncated"].append(has_eos_gathered.mean().item())

        return loss
