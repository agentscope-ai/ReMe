"""Tests for ComponentRegistry."""

# pylint: disable=missing-function-docstring,missing-class-docstring,protected-access,unused-argument

import pytest

from reme4.components.base_component import BaseComponent
from reme4.components.component_registry import ComponentRegistry
from reme4.enumeration import ComponentEnum


class _DummyComponent(BaseComponent):
    component_type = ComponentEnum.FILE_PARSER


class _AnotherComponent(BaseComponent):
    component_type = ComponentEnum.KEYWORD_INDEX


class _NoComponentType:
    pass


class _BaseComponentType(BaseComponent):
    component_type = ComponentEnum.BASE


# -- register & get -----------------------------------------------------------


def test_register_direct_with_explicit_name():
    reg = ComponentRegistry()
    reg.register(_DummyComponent, "my_parser")
    assert reg.get(ComponentEnum.FILE_PARSER, "my_parser") is _DummyComponent


def test_register_direct_defaults_to_class_name():
    reg = ComponentRegistry()
    reg.register(_DummyComponent)
    assert reg.get(ComponentEnum.FILE_PARSER, "_DummyComponent") is _DummyComponent


def test_register_decorator():
    reg = ComponentRegistry()

    @reg.register("alias")
    class MyParser(BaseComponent):
        component_type = ComponentEnum.FILE_PARSER

    assert reg.get(ComponentEnum.FILE_PARSER, "alias") is MyParser


def test_register_overwrite_warns(caplog):
    reg = ComponentRegistry()
    reg.register(_DummyComponent, "dup")
    reg.register(_DummyComponent, "dup")
    assert reg.get(ComponentEnum.FILE_PARSER, "dup") is _DummyComponent


def test_register_rejects_missing_component_type():
    reg = ComponentRegistry()
    with pytest.raises(TypeError, match="ComponentEnum"):
        reg.register(_NoComponentType, "bad")


def test_register_rejects_empty_name():
    reg = ComponentRegistry()
    with pytest.raises(ValueError, match="empty"):
        reg._do_register(_DummyComponent, "")


def test_register_rejects_non_class_non_string():
    reg = ComponentRegistry()
    with pytest.raises(TypeError, match="Expected a class or string"):
        reg.register(42)


# -- get_all ------------------------------------------------------------------


def test_get_all_returns_copy():
    reg = ComponentRegistry()
    reg.register(_DummyComponent, "a")
    reg.register(_AnotherComponent, "b")

    parsers = reg.get_all(ComponentEnum.FILE_PARSER)
    assert parsers == {"a": _DummyComponent}

    indexes = reg.get_all(ComponentEnum.KEYWORD_INDEX)
    assert indexes == {"b": _AnotherComponent}

    # Mutating the copy doesn't affect the registry.
    parsers["hacked"] = _DummyComponent
    assert "hacked" not in reg.get_all(ComponentEnum.FILE_PARSER)


def test_get_all_unknown_type_returns_empty():
    reg = ComponentRegistry()
    assert not reg.get_all(ComponentEnum.LLM)


# -- get (miss) ---------------------------------------------------------------


def test_get_nonexistent_returns_none():
    reg = ComponentRegistry()
    assert reg.get(ComponentEnum.FILE_PARSER, "nope") is None


# -- unregister ---------------------------------------------------------------


def test_unregister_existing():
    reg = ComponentRegistry()
    reg.register(_DummyComponent, "x")
    assert reg.unregister(ComponentEnum.FILE_PARSER, "x") is True
    assert reg.get(ComponentEnum.FILE_PARSER, "x") is None


def test_unregister_missing_returns_false():
    reg = ComponentRegistry()
    assert reg.unregister(ComponentEnum.FILE_PARSER, "nope") is False


# -- clear --------------------------------------------------------------------


def test_clear():
    reg = ComponentRegistry()
    reg.register(_DummyComponent, "a")
    reg.register(_AnotherComponent, "b")
    reg.clear()
    assert not reg.get_all(ComponentEnum.FILE_PARSER)
    assert not reg.get_all(ComponentEnum.KEYWORD_INDEX)


if __name__ == "__main__":
    print("\n=== ComponentRegistry Tests ===")
    test_register_direct_with_explicit_name()
    test_register_direct_defaults_to_class_name()
    test_register_decorator()
    test_register_rejects_missing_component_type()
    test_register_rejects_empty_name()
    test_register_rejects_non_class_non_string()
    test_get_all_returns_copy()
    test_get_all_unknown_type_returns_empty()
    test_get_nonexistent_returns_none()
    test_unregister_existing()
    test_unregister_missing_returns_false()
    test_clear()
    print("\n所有测试通过!")
