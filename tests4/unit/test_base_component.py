"""Tests for BaseComponent, Dependency, and ComponentMixin."""

# pylint: disable=protected-access,missing-function-docstring,missing-class-docstring,attribute-defined-outside-init

import asyncio
import os
import tempfile

import pytest

from reme4.components.base_component import BaseComponent, ComponentMixin, Dependency
from reme4.enumeration import ComponentEnum


# -- Test subclasses ----------------------------------------------------------


class StubComponent(BaseComponent):
    component_type = ComponentEnum.FILE_CHUNKER

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_count = 0
        self.close_count = 0

    async def _start(self):
        self.start_count += 1

    async def _close(self):
        self.close_count += 1


class DepTarget(BaseComponent):
    component_type = ComponentEnum.KEYWORD_INDEX


class RequiredDepTarget(BaseComponent):
    component_type = ComponentEnum.FILE_GRAPH


# -- Dependency ---------------------------------------------------------------


def test_dependency_repr_optional():
    dep = Dependency(ComponentEnum.FILE_CHUNKER, "my_parser", optional=True)
    assert "?" in repr(dep)
    assert "file_chunker" in repr(dep)


def test_dependency_repr_required():
    dep = Dependency(ComponentEnum.FILE_CHUNKER, "my_parser", optional=False)
    assert "?" not in repr(dep)


def test_dependency_getattr_raises():
    dep = Dependency(ComponentEnum.FILE_CHUNKER, "my_parser")
    with pytest.raises(RuntimeError, match="accessed before start"):
        _ = dep.some_method


# -- bind ---------------------------------------------------------------------


def test_bind_returns_none_for_empty_name():
    result = BaseComponent.bind(None, DepTarget)
    assert result is None

    result = BaseComponent.bind("", DepTarget)
    assert result is None


def test_bind_returns_dependency_placeholder():
    result = BaseComponent.bind("my_index", DepTarget)
    assert isinstance(result, Dependency)
    assert result.ctype == ComponentEnum.KEYWORD_INDEX
    assert result.name == "my_index"


def test_bind_rejects_base_component_type():
    class BadTarget(BaseComponent):
        component_type = ComponentEnum.BASE

    with pytest.raises(TypeError, match="non-BASE"):
        BaseComponent.bind("x", BadTarget)


def test_bind_rejects_no_component_type():
    class NoType:
        pass

    with pytest.raises(TypeError, match="non-BASE"):
        BaseComponent.bind("x", NoType)


def test_bind_with_default_factory():
    def factory():
        return DepTarget(name="default")

    result = BaseComponent.bind("idx", DepTarget, default_factory=factory)
    assert isinstance(result, Dependency)
    assert result.default_factory is factory


def test_bind_optional_flag():
    dep = BaseComponent.bind("idx", DepTarget, optional=False)
    assert dep.optional is False


# -- dependencies property ----------------------------------------------------


def test_dependencies_lists_unresolved():
    comp = StubComponent()
    comp.dep1 = Dependency(ComponentEnum.KEYWORD_INDEX, "a")
    comp.dep2 = Dependency(ComponentEnum.FILE_GRAPH, "b")
    comp.normal_attr = "not a dep"
    deps = comp.dependencies
    assert len(deps) == 2


# -- lifecycle ----------------------------------------------------------------


def test_start_close_idempotent():
    async def run():
        comp = StubComponent()
        await comp.start()
        await comp.start()
        assert comp.start_count == 1
        assert comp.is_started is True

        await comp.close()
        await comp.close()
        assert comp.close_count == 1
        assert comp.is_started is False

    asyncio.run(run())


def test_restart():
    async def run():
        comp = StubComponent()
        await comp.start()
        await comp.restart()
        assert comp.start_count == 2
        assert comp.close_count == 1
        assert comp.is_started is True
        await comp.close()

    asyncio.run(run())


def test_async_context_manager():
    async def run():
        comp = StubComponent()
        async with comp as c:
            assert c is comp
            assert comp.is_started is True
        assert comp.is_started is False

    asyncio.run(run())


# -- standalone resolution ----------------------------------------------------


def test_resolve_standalone_optional_becomes_none():
    async def run():
        comp = StubComponent()
        comp.dep = BaseComponent.bind("idx", DepTarget)
        await comp.start()
        assert comp.dep is None
        await comp.close()

    asyncio.run(run())


def test_resolve_standalone_with_default_factory():
    async def run():
        comp = StubComponent()
        comp.dep = BaseComponent.bind(
            "idx",
            DepTarget,
            default_factory=lambda: DepTarget(name="auto"),
        )
        await comp.start()
        assert isinstance(comp.dep, DepTarget)
        assert comp.dep.name == "auto"
        assert comp.dep in comp._owned
        await comp.close()

    asyncio.run(run())


def test_resolve_standalone_required_no_factory_keeps_placeholder():
    async def run():
        comp = StubComponent()
        comp.dep = BaseComponent.bind("idx", DepTarget, optional=False)
        await comp.start()
        assert isinstance(comp.dep, Dependency)
        await comp.close()

    asyncio.run(run())


# -- owned component lifecycle cascade ----------------------------------------


def test_owned_components_started_and_closed():
    async def run():
        owned = StubComponent(name="owned")
        parent = StubComponent(name="parent")
        parent.dep = BaseComponent.bind(
            "sub",
            StubComponent,
            default_factory=lambda: owned,
        )
        await parent.start()
        assert owned.is_started is True

        await parent.close()
        assert owned.is_started is False

    asyncio.run(run())


# -- context-bound resolution -------------------------------------------------


def test_resolve_from_context():
    async def run():
        from reme4.components.application_context import ApplicationContext

        target = DepTarget(name="real_index")
        ctx = ApplicationContext()
        ctx.components = {ComponentEnum.KEYWORD_INDEX: {"real_index": target}}

        comp = StubComponent(app_context=ctx)
        comp.dep = BaseComponent.bind("real_index", DepTarget)
        await comp.start()
        assert comp.dep is target
        await comp.close()

    asyncio.run(run())


def test_resolve_from_context_optional_missing():
    async def run():
        from reme4.components.application_context import ApplicationContext

        ctx = ApplicationContext()
        ctx.components = {}
        comp = StubComponent(app_context=ctx)
        comp.dep = BaseComponent.bind("missing", DepTarget, optional=True)
        await comp.start()
        assert comp.dep is None
        await comp.close()

    asyncio.run(run())


def test_resolve_from_context_required_missing_raises():
    async def run():
        from reme4.components.application_context import ApplicationContext

        ctx = ApplicationContext()
        ctx.components = {}
        comp = StubComponent(app_context=ctx)
        comp.dep = BaseComponent.bind("missing", RequiredDepTarget, optional=False)
        with pytest.raises(ValueError, match="not found"):
            await comp.start()

    asyncio.run(run())


# -- ComponentMixin paths -----------------------------------------------------


def test_vault_path_no_context():
    mixin = ComponentMixin()
    from pathlib import Path

    assert mixin.vault_path == Path.cwd()


def test_to_vault_relative_inside_vault():
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            mixin = ComponentMixin()
            abs_path = mixin.vault_path / "sub" / "file.md"
            rel = mixin.to_vault_relative(abs_path)
            assert rel == str(abs_path.relative_to(mixin.vault_path))
        finally:
            os.chdir(old_cwd)


def test_to_vault_relative_outside_vault():
    mixin = ComponentMixin()
    result = mixin.to_vault_relative("/some/other/path")
    assert result == "/some/other/path"


# -- vault metadata paths -----------------------------------------------------


def test_vault_metadata_path_no_context():
    comp = StubComponent()
    assert comp.vault_metadata_path.name == "metadata"


def test_component_metadata_path():
    comp = StubComponent()
    assert comp.component_metadata_path.name == ComponentEnum.FILE_CHUNKER.value


if __name__ == "__main__":
    print("\n=== BaseComponent Tests ===")
    test_dependency_repr_optional()
    test_dependency_repr_required()
    test_dependency_getattr_raises()
    test_bind_returns_none_for_empty_name()
    test_bind_returns_dependency_placeholder()
    test_bind_rejects_base_component_type()
    test_bind_rejects_no_component_type()
    test_bind_with_default_factory()
    test_bind_optional_flag()
    test_dependencies_lists_unresolved()
    test_start_close_idempotent()
    test_restart()
    test_async_context_manager()
    test_resolve_standalone_optional_becomes_none()
    test_resolve_standalone_with_default_factory()
    test_resolve_standalone_required_no_factory_keeps_placeholder()
    test_owned_components_started_and_closed()
    test_resolve_from_context()
    test_resolve_from_context_optional_missing()
    test_resolve_from_context_required_missing_raises()
    test_vault_path_no_context()
    test_to_vault_relative_outside_vault()
    test_vault_metadata_path_no_context()
    test_component_metadata_path()
    print("\n所有测试通过!")
