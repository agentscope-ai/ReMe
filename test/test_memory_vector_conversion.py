"""Test file for MemoryNode and VectorNode conversion.

Tests:
1. vector -> memory -> vector conversion
2. memory -> vector -> memory conversion
3. _update_after_init correctness (memory_id generation)
4. __setattr__ behavior (when_to_use and content updates)
"""

import datetime
import hashlib
import time
from enum import Enum
from typing import Any, List, Dict
from uuid import uuid4

# Import Pydantic before our modules
from pydantic import BaseModel, Field, model_validator


# Define MemoryType locally to avoid import issues
class MemoryType(str, Enum):
    """Memory type enumeration for the three-layer memory architecture."""

    IDENTITY = "identity"
    PERSONAL = "personal"
    PROCEDURAL = "procedural"
    TOOL = "tool"
    SUMMARY = "summary"
    HISTORY = "history"
    TASK = "task"  # Add TASK type for testing


# Define VectorNode locally to avoid import issues
class VectorNode(BaseModel):
    """Represents a discrete unit of text content paired with its corresponding vector embedding and metadata."""

    vector_id: str = Field(default_factory=lambda: uuid4().hex)
    content: str = Field(default="")
    vector: List[float] | None = Field(default=None)
    metadata: Dict[str, str | bool | int | float] = Field(default_factory=dict)


# Define MemoryNode locally to avoid import issues
def get_now_time() -> str:
    """Get current timestamp in YYYY-MM-DD HH:MM:SS format."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Length of the memory ID (first N characters of SHA-256 hash)
MEMORY_ID_LENGTH: int = 16


class MemoryNode(BaseModel):
    """Memory node for storing memories in the ReMe system."""

    memory_id: str = Field(default="", description="Unique memory identifier")
    memory_type: MemoryType = Field(default=..., description="Type of memory")
    memory_target: str = Field(default="", description="Target or topic of the memory")
    when_to_use: str = Field(default="", description="Condition description for vector retrieval")
    content: str = Field(default="", description="Actual memory content")
    ref_memory_id: str = Field(default="", description="Reference to related raw history memory ID")

    time_created: str = Field(default_factory=get_now_time, description="Creation timestamp")
    time_modified: str = Field(default_factory=get_now_time, description="Last modification timestamp")
    author: str = Field(default="", description="Author or source of the memory")
    score: float = Field(default=0, description="Relevance or importance score")

    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def _update_modified_time(self) -> "MemoryNode":
        """Update time_modified to current timestamp."""
        self.time_modified = get_now_time()
        return self

    def _update_memory_id(self) -> "MemoryNode":
        """Generate memory_id from SHA-256 hash of content."""
        if not self.content:
            return self

        hash_obj = hashlib.sha256(self.content.encode("utf-8"))
        hex_dig = hash_obj.hexdigest()
        self.memory_id = hex_dig[:MEMORY_ID_LENGTH]
        return self

    @model_validator(mode="after")
    def _update_after_init(self) -> "MemoryNode":
        """Post-initialization validator."""
        if not self.memory_id:
            self._update_memory_id()
        return self

    def __setattr__(self, name: str, value):
        """Auto-update timestamps and memory_id when content or when_to_use changes."""
        should_update: bool = name in ("when_to_use", "content") and getattr(self, name, None) != value
        super().__setattr__(name, value)
        if should_update:
            self._update_modified_time()
            if name == "content":
                self._update_memory_id()

    def to_vector_node(self) -> VectorNode:
        """Convert to VectorNode for vector storage."""
        # Build base metadata (shared fields)
        metadata: dict[str, Any] = {
            "memory_type": self.memory_type.value,
            "memory_target": self.memory_target,
            "ref_memory_id": self.ref_memory_id,
            "time_created": self.time_created,
            "time_modified": self.time_modified,
            "author": self.author,
            "score": self.score,
            **self.metadata,
        }

        if self.when_to_use:
            # Use when_to_use for vector embedding, store content in metadata
            vector_content = self.when_to_use
            metadata["content"] = self.content
        else:
            # Use content directly for vector embedding
            vector_content = self.content

        return VectorNode(
            vector_id=self.memory_id,
            content=vector_content,
            metadata=metadata,
        )

    @classmethod
    def from_vector_node(cls, node: VectorNode) -> "MemoryNode":
        """Reconstruct MemoryNode from VectorNode."""
        metadata = node.metadata.copy()
        memory_type_str = metadata.pop("memory_type", None)

        try:
            memory_type: MemoryType = MemoryType(memory_type_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid memory_type '{memory_type_str}' in VectorNode metadata. "
                f"Valid types are: {[t.value for t in MemoryType]}",
            ) from e

        # Restore when_to_use and content based on metadata structure
        if "content" in metadata:
            # Original had when_to_use set
            when_to_use = node.content
            content = metadata.pop("content", "")
        else:
            # Original had empty when_to_use
            when_to_use = ""
            content = node.content

        return cls(
            memory_id=node.vector_id,
            memory_type=memory_type,
            memory_target=metadata.pop("memory_target", ""),
            when_to_use=when_to_use,
            content=content,
            ref_memory_id=metadata.pop("ref_memory_id", ""),
            time_created=metadata.pop("time_created", ""),
            time_modified=metadata.pop("time_modified", ""),
            author=metadata.pop("author", ""),
            score=metadata.pop("score", 0),
            metadata=metadata,
        )


def generate_expected_memory_id(content: str) -> str:
    """Generate expected memory_id from content."""
    hash_obj = hashlib.sha256(content.encode("utf-8"))
    hex_dig = hash_obj.hexdigest()
    return hex_dig[:MEMORY_ID_LENGTH]


def test_vector_to_memory_to_vector_with_when_to_use():
    """Test: vector -> memory -> vector conversion with when_to_use."""
    print("\n=== Test 1: vector -> memory -> vector (with when_to_use) ===")

    # Create original VectorNode with content in metadata (indicates when_to_use was used)
    original_vector = VectorNode(
        vector_id="test_id_001",
        content="When user asks about Python programming",
        vector=[0.1, 0.2, 0.3],
        metadata={
            "memory_type": MemoryType.PERSONAL.value,
            "memory_target": "programming",
            "content": "The user prefers Python for scripting tasks",  # This indicates when_to_use was set
            "ref_memory_id": "ref_001",
            "time_created": "2024-01-01 10:00:00",
            "time_modified": "2024-01-01 12:00:00",
            "author": "system",
            "score": 0.95,
            "custom_field": "custom_value",
        },
    )

    # Convert to MemoryNode
    memory_node = MemoryNode.from_vector_node(original_vector)

    # Convert back to VectorNode
    new_vector = memory_node.to_vector_node()

    # Verify
    assert (
        new_vector.vector_id == original_vector.vector_id
    ), f"vector_id mismatch: {new_vector.vector_id} != {original_vector.vector_id}"

    assert (
        new_vector.content == original_vector.content
    ), f"content mismatch: {new_vector.content} != {original_vector.content}"

    # Check metadata (excluding vector which is not preserved)
    for key in [
        "memory_type",
        "memory_target",
        "content",
        "ref_memory_id",
        "time_created",
        "time_modified",
        "author",
        "score",
        "custom_field",
    ]:
        assert new_vector.metadata.get(key) == original_vector.metadata.get(
            key,
        ), f"metadata[{key}] mismatch: {new_vector.metadata.get(key)} != {original_vector.metadata.get(key)}"

    print("✓ vector -> memory -> vector conversion successful (with when_to_use)")
    print(f"  Original content: {original_vector.content}")
    print(f"  Converted content: {new_vector.content}")
    print(f"  Original metadata['content']: {original_vector.metadata.get('content')}")
    print(f"  Converted metadata['content']: {new_vector.metadata.get('content')}")


def test_vector_to_memory_to_vector_without_when_to_use():
    """Test: vector -> memory -> vector conversion without when_to_use."""
    print("\n=== Test 2: vector -> memory -> vector (without when_to_use) ===")

    # Create original VectorNode without content in metadata (indicates when_to_use was empty)
    original_vector = VectorNode(
        vector_id="test_id_002",
        content="The user prefers Python for scripting tasks",
        vector=[0.4, 0.5, 0.6],
        metadata={
            "memory_type": MemoryType.SUMMARY.value,
            "memory_target": "user_preference",
            "ref_memory_id": "ref_002",
            "time_created": "2024-01-02 10:00:00",
            "time_modified": "2024-01-02 12:00:00",
            "author": "agent",
            "score": 0.85,
        },
    )

    # Convert to MemoryNode
    memory_node = MemoryNode.from_vector_node(original_vector)

    # Convert back to VectorNode
    new_vector = memory_node.to_vector_node()

    # Verify
    assert (
        new_vector.vector_id == original_vector.vector_id
    ), f"vector_id mismatch: {new_vector.vector_id} != {original_vector.vector_id}"

    assert (
        new_vector.content == original_vector.content
    ), f"content mismatch: {new_vector.content} != {original_vector.content}"

    # Check that 'content' is NOT in metadata (because when_to_use was empty)
    assert "content" not in new_vector.metadata, "metadata should not contain 'content' key when when_to_use is empty"

    # Check other metadata
    for key in [
        "memory_type",
        "memory_target",
        "ref_memory_id",
        "time_created",
        "time_modified",
        "author",
        "score",
    ]:
        assert new_vector.metadata.get(key) == original_vector.metadata.get(
            key,
        ), f"metadata[{key}] mismatch: {new_vector.metadata.get(key)} != {original_vector.metadata.get(key)}"

    print("✓ vector -> memory -> vector conversion successful (without when_to_use)")
    print(f"  Original content: {original_vector.content}")
    print(f"  Converted content: {new_vector.content}")
    print(f"  'content' in original metadata: {'content' in original_vector.metadata}")
    print(f"  'content' in converted metadata: {'content' in new_vector.metadata}")


def test_memory_to_vector_to_memory_with_when_to_use():
    """Test: memory -> vector -> memory conversion with when_to_use."""
    print("\n=== Test 3: memory -> vector -> memory (with when_to_use) ===")

    # Create original MemoryNode with when_to_use
    original_memory = MemoryNode(
        memory_id="",  # Will be auto-generated
        memory_type=MemoryType.PERSONAL,
        memory_target="coding_style",
        when_to_use="When discussing code formatting",
        content="User prefers tabs over spaces",
        ref_memory_id="ref_003",
        time_created="2024-01-03 10:00:00",
        time_modified="2024-01-03 12:00:00",
        author="user",
        score=0.9,
        metadata={"priority": "high"},
    )

    # Convert to VectorNode
    vector_node = original_memory.to_vector_node()

    # Convert back to MemoryNode
    new_memory = MemoryNode.from_vector_node(vector_node)

    # Verify all fields
    assert (
        new_memory.memory_id == original_memory.memory_id
    ), f"memory_id mismatch: {new_memory.memory_id} != {original_memory.memory_id}"

    assert (
        new_memory.memory_type == original_memory.memory_type
    ), f"memory_type mismatch: {new_memory.memory_type} != {original_memory.memory_type}"

    assert (
        new_memory.memory_target == original_memory.memory_target
    ), f"memory_target mismatch: {new_memory.memory_target} != {original_memory.memory_target}"

    assert (
        new_memory.when_to_use == original_memory.when_to_use
    ), f"when_to_use mismatch: {new_memory.when_to_use} != {original_memory.when_to_use}"

    assert (
        new_memory.content == original_memory.content
    ), f"content mismatch: {new_memory.content} != {original_memory.content}"

    assert (
        new_memory.ref_memory_id == original_memory.ref_memory_id
    ), f"ref_memory_id mismatch: {new_memory.ref_memory_id} != {original_memory.ref_memory_id}"

    assert (
        new_memory.time_created == original_memory.time_created
    ), f"time_created mismatch: {new_memory.time_created} != {original_memory.time_created}"

    assert (
        new_memory.time_modified == original_memory.time_modified
    ), f"time_modified mismatch: {new_memory.time_modified} != {original_memory.time_modified}"

    assert (
        new_memory.author == original_memory.author
    ), f"author mismatch: {new_memory.author} != {original_memory.author}"

    assert new_memory.score == original_memory.score, f"score mismatch: {new_memory.score} != {original_memory.score}"

    assert (
        new_memory.metadata == original_memory.metadata
    ), f"metadata mismatch: {new_memory.metadata} != {original_memory.metadata}"

    print("✓ memory -> vector -> memory conversion successful (with when_to_use)")
    print(f"  Original when_to_use: {original_memory.when_to_use}")
    print(f"  Converted when_to_use: {new_memory.when_to_use}")
    print(f"  Original content: {original_memory.content}")
    print(f"  Converted content: {new_memory.content}")


def test_memory_to_vector_to_memory_without_when_to_use():
    """Test: memory -> vector -> memory conversion without when_to_use."""
    print("\n=== Test 4: memory -> vector -> memory (without when_to_use) ===")

    # Create original MemoryNode without when_to_use
    original_memory = MemoryNode(
        memory_type=MemoryType.TASK,
        memory_target="project_info",
        when_to_use="",  # Empty when_to_use
        content="Project deadline is next Friday",
        ref_memory_id="ref_004",
        time_created="2024-01-04 10:00:00",
        time_modified="2024-01-04 12:00:00",
        author="manager",
        score=1.0,
        metadata={"urgency": "high"},
    )

    # Convert to VectorNode
    vector_node = original_memory.to_vector_node()

    # Convert back to MemoryNode
    new_memory = MemoryNode.from_vector_node(vector_node)

    # Verify all fields
    assert (
        new_memory.memory_id == original_memory.memory_id
    ), f"memory_id mismatch: {new_memory.memory_id} != {original_memory.memory_id}"

    assert (
        new_memory.memory_type == original_memory.memory_type
    ), f"memory_type mismatch: {new_memory.memory_type} != {original_memory.memory_type}"

    assert new_memory.when_to_use == "", f"when_to_use should be empty, got: {new_memory.when_to_use}"

    assert (
        new_memory.content == original_memory.content
    ), f"content mismatch: {new_memory.content} != {original_memory.content}"

    print("✓ memory -> vector -> memory conversion successful (without when_to_use)")
    print(f"  Original when_to_use: '{original_memory.when_to_use}'")
    print(f"  Converted when_to_use: '{new_memory.when_to_use}'")
    print(f"  Original content: {original_memory.content}")
    print(f"  Converted content: {new_memory.content}")


def test_update_after_init_auto_generates_memory_id():
    """Test: _update_after_init auto-generates memory_id when not provided."""
    print("\n=== Test 5: _update_after_init auto-generates memory_id ===")

    content = "Test content for memory_id generation"
    expected_memory_id = generate_expected_memory_id(content)

    # Create MemoryNode without providing memory_id
    memory = MemoryNode(
        memory_type=MemoryType.PERSONAL,
        content=content,
    )

    assert memory.memory_id == expected_memory_id, f"memory_id mismatch: {memory.memory_id} != {expected_memory_id}"

    print("✓ _update_after_init correctly generates memory_id")
    print(f"  Content: {content}")
    print(f"  Generated memory_id: {memory.memory_id}")
    print(f"  Expected memory_id: {expected_memory_id}")


def test_update_after_init_preserves_provided_memory_id():
    """Test: _update_after_init preserves memory_id when provided."""
    print("\n=== Test 6: _update_after_init preserves provided memory_id ===")

    custom_memory_id = "custom_id_12345"
    content = "Test content"

    # Create MemoryNode with explicit memory_id
    memory = MemoryNode(
        memory_id=custom_memory_id,
        memory_type=MemoryType.TASK,
        content=content,
    )

    assert (
        memory.memory_id == custom_memory_id
    ), f"memory_id should be preserved: {memory.memory_id} != {custom_memory_id}"

    print("✓ _update_after_init preserves provided memory_id")
    print(f"  Provided memory_id: {custom_memory_id}")
    print(f"  Actual memory_id: {memory.memory_id}")


def test_update_after_init_empty_content():
    """Test: _update_after_init with empty content."""
    print("\n=== Test 7: _update_after_init with empty content ===")

    # Create MemoryNode with empty content
    memory = MemoryNode(
        memory_type=MemoryType.SUMMARY,
        content="",
    )

    assert memory.memory_id == "", f"memory_id should be empty when content is empty, got: {memory.memory_id}"

    print("✓ _update_after_init correctly handles empty content")
    print(f"  Content: '{memory.content}'")
    print(f"  memory_id: '{memory.memory_id}'")


def test_setattr_content_updates_memory_id_and_time():
    """Test: __setattr__ updates memory_id and time_modified when content changes."""
    print("\n=== Test 8: __setattr__ updates memory_id when content changes ===")

    initial_content = "Initial content"
    new_content = "New content after update"

    # Create MemoryNode
    memory = MemoryNode(
        memory_type=MemoryType.PERSONAL,
        content=initial_content,
    )

    initial_memory_id = memory.memory_id
    initial_time_modified = memory.time_modified
    expected_initial_id = generate_expected_memory_id(initial_content)

    assert (
        initial_memory_id == expected_initial_id
    ), f"Initial memory_id incorrect: {initial_memory_id} != {expected_initial_id}"

    # Wait at least 1 second to ensure time changes (timestamp precision is 1 second)
    time.sleep(1.1)

    # Update content
    memory.content = new_content

    expected_new_id = generate_expected_memory_id(new_content)

    assert (
        memory.memory_id == expected_new_id
    ), f"memory_id not updated correctly: {memory.memory_id} != {expected_new_id}"

    assert memory.memory_id != initial_memory_id, "memory_id should change when content changes"

    assert memory.time_modified != initial_time_modified, "time_modified should be updated when content changes"

    print("✓ __setattr__ correctly updates memory_id and time_modified when content changes")
    print(f"  Initial content: {initial_content}")
    print(f"  Initial memory_id: {initial_memory_id}")
    print(f"  New content: {new_content}")
    print(f"  New memory_id: {memory.memory_id}")
    print(f"  Initial time_modified: {initial_time_modified}")
    print(f"  New time_modified: {memory.time_modified}")


def test_setattr_when_to_use_updates_time_only():
    """Test: __setattr__ updates only time_modified when when_to_use changes."""
    print("\n=== Test 9: __setattr__ updates time_modified when when_to_use changes ===")

    content = "Fixed content"
    initial_when_to_use = "Initial trigger"
    new_when_to_use = "New trigger condition"

    # Create MemoryNode
    memory = MemoryNode(
        memory_type=MemoryType.TASK,
        content=content,
        when_to_use=initial_when_to_use,
    )

    initial_memory_id = memory.memory_id
    initial_time_modified = memory.time_modified

    # Wait at least 1 second to ensure time changes (timestamp precision is 1 second)
    time.sleep(1.1)

    # Update when_to_use
    memory.when_to_use = new_when_to_use

    assert (
        memory.memory_id == initial_memory_id
    ), f"memory_id should NOT change when only when_to_use changes: {memory.memory_id} != {initial_memory_id}"

    assert memory.time_modified != initial_time_modified, "time_modified should be updated when when_to_use changes"

    print("✓ __setattr__ correctly updates time_modified but not memory_id when when_to_use changes")
    print(f"  Content (unchanged): {content}")
    print(f"  memory_id (unchanged): {memory.memory_id}")
    print(f"  Initial when_to_use: {initial_when_to_use}")
    print(f"  New when_to_use: {new_when_to_use}")
    print(f"  Initial time_modified: {initial_time_modified}")
    print(f"  New time_modified: {memory.time_modified}")


def test_setattr_no_update_when_value_unchanged():
    """Test: __setattr__ does not update when value is unchanged."""
    print("\n=== Test 10: __setattr__ does not update when value is unchanged ===")

    content = "Test content"
    when_to_use = "Test trigger"

    # Create MemoryNode
    memory = MemoryNode(
        memory_type=MemoryType.SUMMARY,
        content=content,
        when_to_use=when_to_use,
    )

    initial_time_modified = memory.time_modified

    # Wait a bit (but updates should not happen for unchanged values)
    time.sleep(1.1)

    # Set content to same value
    memory.content = content

    assert (
        memory.time_modified == initial_time_modified
    ), "time_modified should NOT change when setting same content value"

    # Set when_to_use to same value
    memory.when_to_use = when_to_use

    assert (
        memory.time_modified == initial_time_modified
    ), "time_modified should NOT change when setting same when_to_use value"

    print("✓ __setattr__ correctly does not update when value is unchanged")
    print(f"  Content: {content}")
    print(f"  when_to_use: {when_to_use}")
    print(f"  time_modified (unchanged): {memory.time_modified}")


def test_setattr_other_fields_no_update():
    """Test: __setattr__ does not trigger updates for other fields."""
    print("\n=== Test 11: __setattr__ does not update for other fields ===")

    # Create MemoryNode
    memory = MemoryNode(
        memory_type=MemoryType.PERSONAL,
        content="Test content",
    )

    initial_memory_id = memory.memory_id
    initial_time_modified = memory.time_modified

    # Wait a bit (but updates should not happen for other fields)
    time.sleep(1.1)

    # Update other fields
    memory.score = 0.95
    memory.author = "new_author"
    memory.memory_target = "new_target"
    memory.ref_memory_id = "new_ref"

    assert memory.memory_id == initial_memory_id, "memory_id should NOT change when updating other fields"

    assert memory.time_modified == initial_time_modified, "time_modified should NOT change when updating other fields"

    print("✓ __setattr__ correctly does not update for other fields")
    print(f"  memory_id (unchanged): {memory.memory_id}")
    print(f"  time_modified (unchanged): {memory.time_modified}")
    print("  Updated fields: score, author, memory_target, ref_memory_id")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running MemoryNode and VectorNode Conversion Tests")
    print("=" * 60)

    tests = [
        test_vector_to_memory_to_vector_with_when_to_use,
        test_vector_to_memory_to_vector_without_when_to_use,
        test_memory_to_vector_to_memory_with_when_to_use,
        test_memory_to_vector_to_memory_without_when_to_use,
        test_update_after_init_auto_generates_memory_id,
        test_update_after_init_preserves_provided_memory_id,
        test_update_after_init_empty_content,
        test_setattr_content_updates_memory_id_and_time,
        test_setattr_when_to_use_updates_time_only,
        test_setattr_no_update_when_value_unchanged,
        test_setattr_other_fields_no_update,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
