"""
Unit tests for BaseOp and operator composition (>>, <<, |).
Tests asynchronous execution mode.
"""

import asyncio

from reme_ai.core.op import BaseOp
from reme_ai.core.schema import ToolCall, ToolAttr


class AddOp(BaseOp):
    """Simple operator that adds a value to a number in context."""

    def __init__(self, value: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "name": self.name,
                "description": f"Add {self.value} to input",
                "parameters": ToolAttr(
                    **{
                        "type": "object",
                        "properties": {
                            "number": {"type": "integer", "description": "Input number"},
                        },
                        "required": ["number"],
                    },
                ),
            },
        )

    async def execute(self):
        """Async execution: add value to input number."""
        self.context["number"] += self.value
        self.output = self.context["number"]


class MultiplyOp(BaseOp):
    """Simple operator that multiplies a number in context."""

    def __init__(self, factor: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "name": self.name,
                "description": f"Multiply by {self.factor}",
                "parameters": ToolAttr(
                    **{
                        "type": "object",
                        "properties": {
                            "number": {"type": "integer", "description": "Input number"},
                        },
                        "required": ["number"],
                    },
                ),
            },
        )

    async def execute(self):
        """Async execution: multiply input number."""
        self.context["number"] *= self.factor
        self.output = self.context["number"]


class AppendOp(BaseOp):
    """Operator that appends a value to a list in context."""

    def __init__(self, value: str = "", **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def _build_tool_call(self) -> ToolCall:
        return ToolCall(
            **{
                "name": self.name,
                "description": f"Append {self.value} to list",
                "parameters": ToolAttr(
                    **{
                        "type": "object",
                        "properties": {
                            "items": {"type": "array", "description": "List of items"},
                        },
                        "required": ["items"],
                    },
                ),
            },
        )

    async def execute(self):
        """Async execution: append value to list."""
        self.context["items"].append(self.value)
        self.output = self.context["items"]


async def test_basic_async_call():
    """Test basic asynchronous operator execution."""
    op = AddOp(value=5, name="add_5")
    await op.call(number=10)
    number = op.context["number"]
    assert number == 15, f"Expected context result 15, got {number}"
    print("✓ test_basic_async_call passed")


async def test_sequential_composition_async():
    """Test >> operator for sequential composition in async mode."""
    add_op = AddOp(value=5, name="add_5")
    multiply_op = MultiplyOp(factor=2, name="multiply_2")
    composed = add_op >> multiply_op
    await composed.call(number=10)

    # (10 + 5) * 2 = 30
    assert composed.context["number"] == 30, f"Expected 30, got {composed.context['number']}"
    print("✓ test_sequential_composition_async passed")


async def test_parallel_composition_async():
    """Test | operator for parallel composition in async mode."""
    append_a = AppendOp(value="A", name="append_a")
    append_b = AppendOp(value="B", name="append_b")
    append_c = AppendOp(value="C", name="append_c")

    composed = append_a | append_b | append_c

    await composed.call(items=[])

    # All should append to the list
    items = composed.context["items"]
    assert len(items) == 3, f"Expected 3 items, got {len(items)}"
    assert set(items) == {"A", "B", "C"}, f"Expected A,B,C, got {items}"
    print("✓ test_parallel_composition_async passed")


async def test_add_sub_ops_async():
    """Test << operator for adding sub-operations in async mode."""
    parent_op = BaseOp(name="parent")
    child1 = AddOp(value=5, name="child1")
    child2 = MultiplyOp(factor=2, name="child2")

    _ = parent_op << child1
    _ = parent_op << child2

    assert len(parent_op.sub_ops) == 2, f"Expected 2 sub_ops, got {len(parent_op.sub_ops)}"
    sub_op_names = [op.name for op in parent_op.sub_ops]
    assert "child1" in sub_op_names, "child1 not in sub_ops"
    assert "child2" in sub_op_names, "child2 not in sub_ops"
    print("✓ test_add_sub_ops_async passed")


async def test_add_sub_ops_dict():
    """Test << operator with dictionary of operations."""
    parent_op = BaseOp(name="parent")
    ops_dict = {
        "add": AddOp(value=5, name="add"),
        "multiply": MultiplyOp(factor=2, name="multiply"),
    }

    _ = parent_op << ops_dict

    assert len(parent_op.sub_ops) == 2, f"Expected 2 ops_dict, got {len(parent_op.sub_ops)}"
    sub_op_names = [op.name for op in parent_op.sub_ops]
    assert "add" in sub_op_names, "add not in ops_dict"
    assert "multiply" in sub_op_names, "multiply not in ops_dict"
    print("✓ test_add_sub_ops_dict passed")


async def test_add_sub_ops_list():
    """Test << operator with list of operations."""
    parent_op = BaseOp(name="parent")
    sub_ops = [
        AddOp(value=5, name="add"),
        MultiplyOp(factor=2, name="multiply"),
    ]

    _ = parent_op << sub_ops

    assert len(parent_op.sub_ops) == 2, f"Expected 2 sub_ops, got {len(parent_op.sub_ops)}"
    sub_op_names = [op.name for op in parent_op.sub_ops]
    assert "add" in sub_op_names, "add not in sub_ops"
    assert "multiply" in sub_op_names, "multiply not in sub_ops"
    print("✓ test_add_sub_ops_list passed")


async def test_mixed_composition_async():
    """Test mixing >> and | operators in async mode."""
    # (add_5 >> multiply_2) | (add_10 >> multiply_3)
    seq1 = AddOp(value=5, name="add_5") >> MultiplyOp(factor=2, name="multiply_2")
    seq2 = AddOp(value=10, name="add_10") >> MultiplyOp(factor=3, name="multiply_3")

    composed = seq1 | seq2

    await composed.call(number=10)

    # Both sequences execute in parallel with shared context
    # seq1: (10 + 5) * 2 = 30
    # seq2: (30 + 10) * 3 = 120 (builds on seq1's result due to shared context)
    # The exact result depends on execution order and timing
    # With current implementation, result is 120
    assert composed.context["number"] == 120, f"Expected 120, got {composed.context['number']}"
    print("✓ test_mixed_composition_async passed")


async def test_op_copy():
    """Test operator copy functionality."""
    original = AddOp(value=5, name="original")
    copy_op = original.copy(name="copy")

    assert copy_op.name == "copy", f"Expected name 'copy', got {copy_op.name}"
    assert copy_op.value == 5, f"Expected value 5, got {copy_op.value}"
    assert copy_op is not original, "Copy should be a different object"
    print("✓ test_op_copy passed")


async def test_input_mapping():
    """Test input_mapping parameter."""
    op = AddOp(
        value=5,
        name="add_5",
        input_mapping={"x": "number"},  # Map x to number
    )

    await op.call(x=10)  # Input is 'x' not 'number'

    assert op.context["number"] == 15, f"Expected number=15, got {op.context['number']}"
    print("✓ test_input_mapping passed")


async def test_output_mapping():
    """Test output_mapping parameter."""
    op = AddOp(
        value=5,
        name="add_5",
        output_mapping={"number": "final_result"},  # Map number to final_result
    )

    await op.call(number=10)

    assert op.context["number"] == 15, f"Expected number=15, got {op.context['number']}"
    assert op.context["final_result"] == 15, f"Expected final_result=15, got {op.context['final_result']}"
    print("✓ test_output_mapping passed")


async def test_validation_missing_required():
    """Test that missing required inputs raise an error."""
    op = AddOp(value=5, name="add_5", raise_exception=True)

    try:
        await op.call()  # Missing 'number' field
        assert False, "Should have raised ValueError for missing required input"
    except ValueError as e:
        assert "number" in str(e), f"Expected error about 'number', got: {e}"
        print("✓ test_validation_missing_required passed")


async def test_max_retries():
    """Test max_retries parameter with failing operation."""

    class FailingOp(BaseOp):
        """An operation that always fails."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.attempt_count = 0

        def _build_tool_call(self) -> ToolCall:
            return ToolCall(
                **{
                    "name": self.name,
                    "description": "Always fails",
                    "parameters": ToolAttr(**{"type": "object", "properties": {}}),
                    "output": ToolAttr(
                        **{
                            "type": "object",
                            "properties": {
                                "result": ToolAttr(**{"type": "string", "description": "Result"}),
                            },
                        },
                    ),
                },
            )

        async def execute(self):
            self.attempt_count += 1
            raise RuntimeError(f"Attempt {self.attempt_count} failed")

    op = FailingOp(max_retries=3, name="failing")

    await op.call()

    assert op.attempt_count == 3, f"Expected 3 attempts, got {op.attempt_count}"
    print("✓ test_max_retries passed")


async def async_main():
    """Run all async tests."""
    await test_basic_async_call()
    await test_sequential_composition_async()
    await test_parallel_composition_async()
    await test_add_sub_ops_async()
    await test_add_sub_ops_dict()
    await test_add_sub_ops_list()
    await test_mixed_composition_async()
    await test_op_copy()
    await test_input_mapping()
    await test_output_mapping()
    await test_validation_missing_required()
    await test_max_retries()


if __name__ == "__main__":
    print("Running BaseOp composition tests...\n")

    # Async tests
    print("=== Asynchronous Tests ===")
    asyncio.run(async_main())

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
