# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CountdownHintInteraction(BaseInteraction):
    """Interaction handler for countdown task with hint support.

    During RL rollouts, when the model outputs <request></request>, this handler
    provides the next intermediate expression hint from the ground truth.

    Supports both sequential and smart hint selection via HintSelector.

    Flow:
    1. Model generates: <think>reasoning...</think><request></request>
    2. System responds: <response>(35 / 1)</response>
    3. Model continues: <think>more reasoning...</think><answer>...</answer>

    The reward function penalizes hint usage via hint_penalty.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}
        self.request_tag_pattern = re.compile(r"<request>.*?</request>|<request>|<request/>", re.DOTALL)

        # Smart hint selection
        self.hint_selector = None
        hint_selection = config.get("hint_selection", "sequential")
        if hint_selection == "smart":
            from pipeline.core.hint_selector import HintSelector
            helper_model = config.get("helper_model")
            if helper_model:
                self.hint_selector = HintSelector(
                    strategy="smart",
                    helper_model=helper_model,
                    tensor_parallel_size=config.get("helper_tensor_parallel_size", 1),
                    gpu_memory_utilization=config.get("helper_gpu_memory_utilization", 0.9),
                )
                logger.info(f"Smart hint selection enabled (helper: {helper_model})")

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Initialize interaction state for a trajectory.

        Args:
            instance_id: Unique ID for this trajectory
            ground_truth: Dict containing 'hints_expr' list of intermediate expressions

        Returns:
            The instance_id
        """
        if instance_id is None:
            instance_id = str(uuid4())

        hints_expr = []
        if ground_truth is not None:
            # Handle both list and string representations
            # Support both hint_exprs (pipeline) and hints_expr (legacy)
            raw_hints = ground_truth.get("hint_exprs", ground_truth.get("hints_expr", []))
            if isinstance(raw_hints, str):
                # Parse string representation like "['(35 / 1)', '(30 + 2)']"
                try:
                    import ast
                    hints_expr = ast.literal_eval(raw_hints)
                except (ValueError, SyntaxError):
                    hints_expr = []
            elif isinstance(raw_hints, list):
                hints_expr = list(raw_hints)

        self._instance_dict[instance_id] = {
            "hints_expr": hints_expr,
            "last_given_index": -1,
            "num_hints_given": 0,
            "ground_truth": ground_truth,
        }

        logger.debug(f"Started countdown hint interaction {instance_id} with {len(hints_expr)} hints")
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]],
        **kwargs,
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Process model output and provide hint if requested.

        Args:
            instance_id: The trajectory ID
            messages: Conversation history

        Returns:
            Tuple of:
            - should_terminate: True if no hint requested (let model finish)
            - response_content: The hint response or empty string
            - turn_score: Always 0.0 (final reward computed by reward function)
            - metadata: Additional info including hint count
        """
        if instance_id not in self._instance_dict:
            logger.warning(f"Unknown instance_id: {instance_id}")
            return True, "", 0.0, {}

        inst = self._instance_dict[instance_id]

        # Get the last assistant message
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_content = msg.get("content", "")
                break

        # Check if model requested a hint
        if not self.request_tag_pattern.search(last_content):
            # No hint requested - let the model continue/finish
            return True, "", 0.0, {"num_hints": inst["num_hints_given"]}

        # Model requested a hint
        hints = inst["hints_expr"]
        last_given = inst["last_given_index"]

        if last_given + 1 < len(hints):
            # Select hint (smart or sequential)
            if self.hint_selector is not None:
                hint_text, new_last = self.hint_selector.select_hint_sync(
                    last_content, hints, last_given,
                )
                if hint_text is None:
                    # Fallback
                    response = "<response>No more hints available.</response>"
                    return False, response, 0.0, {"num_hints": inst["num_hints_given"], "hint_exhausted": True}
            else:
                next_idx = last_given + 1
                hint_text, new_last = hints[next_idx], next_idx

            inst["last_given_index"] = new_last
            inst["num_hints_given"] += 1

            response = f"<response>{hint_text}</response>"
            logger.debug(f"Providing hint (last_given={new_last}): {hint_text}")

            # Continue the interaction (model should keep reasoning)
            return False, response, 0.0, {"num_hints": inst["num_hints_given"], "hint_provided": hint_text}
        else:
            # No more hints available
            response = "<response>No more hints available.</response>"
            logger.debug(f"No more hints available (last_given={last_given}, have {len(hints)})")

            # Continue but with warning
            return False, response, 0.0, {"num_hints": inst["num_hints_given"], "hint_exhausted": True}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        """Calculate score for this interaction.

        Note: The actual reward is computed by the reward function which
        applies hint_penalty. This returns 0.0 as a placeholder.
        """
        if instance_id not in self._instance_dict:
            return 0.0
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Clean up interaction state."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
