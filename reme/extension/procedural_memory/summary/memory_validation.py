"""Memory validation operation for task memory quality control.

This module provides operations to validate the quality of extracted task
memories using LLM-based evaluation, ensuring only high-quality memories
are stored.
"""

import json
import re
from typing import List, Dict, Any

from loguru import logger

from ....core.enumeration import Role
from ....core.op import BaseOp
from ....core.schema.message import Message
from ....core.schema.memory_node import MemoryNode


class MemoryValidation(BaseOp):
    """Validate quality of extracted task memories.

    This operation uses LLM-based evaluation to assess the quality of extracted
    task memories, filtering out low-quality or invalid memories based on
    validation scores and criteria.
    """

    async def execute(self):
        """Validate quality of extracted task memories"""

        task_memories: List[MemoryNode] = []
        task_memories.extend(self.context.get("success_task_memories", []))
        task_memories.extend(self.context.get("failure_task_memories", []))
        task_memories.extend(self.context.get("comparative_task_memories", []))

        if not task_memories:
            logger.info("No task memories found for validation")
            return

        logger.info(f"Validating {len(task_memories)} extracted task memories")

        # Validate task memories
        validated_task_memories = []

        for task_memory in task_memories:
            validation_result = await self._validate_single_task_memory(task_memory)
            if validation_result and validation_result.get("is_valid", False):
                task_memory.score = validation_result.get("score", 0.0)
                validated_task_memories.append(task_memory)
            else:
                reason = validation_result.get("reason", "Unknown reason") if validation_result else "Validation failed"
                logger.warning(f"Task memory validation failed: {reason}")

        logger.info(f"Validated {len(validated_task_memories)} out of {len(task_memories)} task memories")

        # Update context
        self.context.response.answer = json.dumps([x.model_dump() for x in validated_task_memories])
        self.context.response.metadata["memory_list"] = validated_task_memories

    async def _validate_single_task_memory(self, task_memory: MemoryNode) -> Dict[str, Any]:
        """Validate single task memory"""
        validation_info = await self._llm_validate_task_memory(task_memory)
        logger.info(f"Validating: {validation_info}")
        return validation_info

    async def _llm_validate_task_memory(self, task_memory: MemoryNode) -> Dict[str, Any]:
        """Validate task memory using LLM"""
        try:
            prompt = self.prompt_format(
                prompt_name="task_memory_validation_prompt",
                condition=task_memory.when_to_use,
                task_memory_content=task_memory.content,
            )

            def parse_validation(message: Message) -> Dict[str, Any]:
                try:
                    response_content = message.content

                    # Parse validation result
                    # Extract JSON blocks
                    json_pattern = r"```json\s*([\s\S]*?)\s*```"
                    json_blocks = re.findall(json_pattern, response_content)

                    parsed: Dict[str, Any] = {}
                    if json_blocks:
                        raw_json = json_blocks[0]
                        try:
                            parsed = json.loads(raw_json)
                        except json.JSONDecodeError as json_err:
                            logger.warning(
                                f"JSONDecodeError in task_memory_validation, fallback to regex parse: {json_err}",
                            )
                            is_valid_match = re.search(r'"is_valid"\s*:\s*(true|false)', raw_json, re.IGNORECASE)
                            score_match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw_json)

                            if is_valid_match:
                                parsed["is_valid"] = is_valid_match.group(1).lower() == "true"
                            if score_match:
                                parsed["score"] = float(score_match.group(1))

                    is_valid = parsed.get("is_valid", True)
                    score = parsed.get("score", 0.5)

                    # Set validation threshold
                    validation_threshold = self.context.get("validation_threshold", 0.5)

                    return {
                        "is_valid": is_valid and score >= validation_threshold,
                        "score": score,
                        "feedback": response_content,
                        "reason": (
                            ""
                            if (is_valid and score >= validation_threshold)
                            else f"Low validation score ({score:.2f}) or marked as invalid"
                        ),
                    }

                except Exception as e_inner:
                    logger.exception(f"Error parsing validation response: {e_inner}")
                    return {
                        "is_valid": False,
                        "score": 0.0,
                        "feedback": "",
                        "reason": f"Parse error: {str(e_inner)}",
                    }

            return await self.llm.chat(
                messages=[Message(role=Role.USER, content=prompt)],
                callback_fn=parse_validation,
            )

        except Exception as e:
            logger.error(f"LLM validation failed: {e}")
            return {
                "is_valid": False,
                "score": 0.0,
                "feedback": "",
                "reason": f"LLM validation error: {str(e)}",
            }
