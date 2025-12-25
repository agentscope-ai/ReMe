import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reme_ai.core.context.base_context import BaseContext


def test_attribute_access():
    c = BaseContext()
    c.xxx = 123
    assert c.xxx == 123
    assert c["xxx"] == 123


def test_dict_access():
    c = BaseContext()
    c["yyy"] = 456
    assert c.yyy == 456
    assert c["yyy"] == 456


def test_delete_attribute():
    c = BaseContext()
    c.zzz = 789
    del c.zzz
    assert "zzz" not in c


def test_attribute_error():
    c = BaseContext()
    try:
        _ = c.nonexistent
        assert False, "Should raise AttributeError"
    except AttributeError as e:
        assert "nonexistent" in str(e)


def test_pickling():
    c = BaseContext()
    c.foo = "bar"
    c.num = 42

    pickled = pickle.dumps(c)
    restored = pickle.loads(pickled)

    assert restored.foo == "bar"
    assert restored.num == 42
    assert isinstance(restored, BaseContext)


def test_init_with_data():
    c = BaseContext({"a": 1, "b": 2})
    assert c.a == 1
    assert c.b == 2


if __name__ == "__main__":
    test_attribute_access()
    test_dict_access()
    test_delete_attribute()
    test_attribute_error()
    test_pickling()
    test_init_with_data()
    print("All tests passed!")

