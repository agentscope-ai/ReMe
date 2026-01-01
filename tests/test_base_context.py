"""
Unit tests for the BaseContext class in reme_ai.core.context.
Ensures attribute-style and dict-style access work interchangeably.
"""

import pickle
from reme_ai.core.context import BaseContext


def test_attribute_access():
    """Test setting values via attributes and retrieving via items."""
    context = BaseContext()
    context.xxx = 123
    assert context.xxx == 123
    assert context["xxx"] == 123


def test_dict_access():
    """Test setting values via items and retrieving via attributes."""
    context = BaseContext()
    context["yyy"] = 456
    assert context.yyy == 456
    assert context["yyy"] == 456


def test_delete_attribute():
    """Test that deleting an attribute removes it from the internal state."""
    context = BaseContext()
    context.zzz = 789
    del context.zzz
    assert "zzz" not in context


def test_attribute_error():
    """Test that accessing non-existent attributes raises the correct error."""
    context = BaseContext()
    try:
        _ = context.nonexistent
        assert False, "Should raise AttributeError"
    except AttributeError as error:
        assert "nonexistent" in str(error)


def test_pickling():
    """Test that BaseContext instances can be serialized and deserialized."""
    context = BaseContext()
    context.test_value = "bar"
    context.num = 42

    pickled = pickle.dumps(context)
    restored = pickle.loads(pickled)

    assert restored.test_value == "bar"
    assert restored.num == 42
    assert isinstance(restored, BaseContext)


def test_init_with_data():
    """Test that the constructor correctly handles initial dictionary data."""
    context = BaseContext({"a": 1, "b": 2})
    assert context.a == 1
    assert context.b == 2


if __name__ == "__main__":
    test_attribute_access()
    test_dict_access()
    test_delete_attribute()
    test_attribute_error()
    test_pickling()
    test_init_with_data()
    print("All tests passed!")
