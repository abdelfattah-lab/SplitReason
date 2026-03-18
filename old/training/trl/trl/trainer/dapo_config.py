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

from dataclasses import dataclass, field
from typing import Optional

from .grpo_config import GRPOConfig


@dataclass
class DAPOConfig(GRPOConfig):
    r"""
    Configuration class for the [`DAPOTrainer`].

    Subclass of [`GRPOConfig`] with DAPO-specific defaults from the
    [DAPO paper](https://huggingface.co/papers/2503.14476):

    - **Clip-Higher**: `epsilon_high = 0.28`
    - **No reward scaling**: `scale_rewards = False`
    - **Overlong filtering**: `mask_truncated_completions = True`
    - **No KL penalty**: `beta = 0.0` (ref model not loaded)
    - **Token-level global normalization**: `loss_type = "dapo"`
    """

    epsilon_high: Optional[float] = field(
        default=0.28,
        metadata={
            "help": "Upper-bound epsilon for clipping (Clip-Higher). DAPO recommends 0.28."
        },
    )
    scale_rewards: bool = field(
        default=False,
        metadata={
            "help": "Whether to scale rewards by std. DAPO disables this (Dr. GRPO style)."
        },
    )
    mask_truncated_completions: bool = field(
        default=True,
        metadata={
            "help": "Zero out loss for completions that were truncated (no EOS token). "
            "This is the 'overlong filtering' from DAPO."
        },
    )
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. DAPO sets this to 0.0 (no reference model)."
        },
    )
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "Loss normalization type. 'dapo' uses token-level global normalization "
            "across all processes. 'grpo' uses per-sequence normalization (default GRPOTrainer behaviour)."
        },
    )
