# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from __future__ import annotations

from typing import Any, Optional

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)


class ThinkingTool(BaseTool):
    """Lightweight scratchpad tool for internal notes."""

    def __init__(self, config: dict[str, Any], tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        schema = tool_schema or self._build_schema(config)
        super().__init__(config, schema)

    def _build_schema(self, config: dict[str, Any]) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name=config.get("function_name", "thinking"),
                description=(
                    "Maintain a short private planning note. Use it to clarify the current goal, capture your "
                    "read of the situation, and choose the single most useful next step. Internal guidance only."
                ),
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "goal": OpenAIFunctionPropertySchema(
                            type="string",
                            description=(
                                "Immediate sub-goal you are pursuing for the user "
                                '(e.g. "confirm factual support for claim").'
                            ),
                        ),
                        "status": OpenAIFunctionPropertySchema(
                            type="string",
                            description=(
                                "Current read of the situation in 1-2 sentences, including uncertainties."
                            ),
                        ),
                        "next_step": OpenAIFunctionPropertySchema(
                            type="string",
                            description=(
                                'The single next action you will take (e.g. "call search with queries [...]").'
                            ),
                        ),
                    },
                    required=["goal", "status", "next_step"],
                ),
            ),
        )

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        **_: Any,
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        goal = parameters.get("goal") or ""
        status = parameters.get("status") or ""
        next_step = parameters.get("next_step") or ""
        metrics = {
            "thinking/goal_chars": len(goal),
            "thinking/status_chars": len(status),
            "thinking/next_step_chars": len(next_step),
        }
        # Respond tersely to keep the notes private.
        return ToolResponse(text="NOTE_ACCEPTED_PROCEED_WITH_NEXT_STEP"), 0.0, metrics
