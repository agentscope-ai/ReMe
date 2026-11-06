"""Health and memory reports must describe the request's actual workspace."""

import pytest

from reme.application import Application
from reme.components.application_context import ApplicationContext
from reme.components.base_component import BaseComponent
from reme.enumeration import ComponentEnum
from reme.schema import FileNode
from reme.steps.common.health_check import HealthCheckStep
from reme.steps.common.status import _collect_memory


@pytest.mark.asyncio
async def test_health_job_reports_beta_failure_without_affecting_alpha(tmp_path):
    """Exercise real graphs and the public job before and after only beta stops."""
    app = Application(
        workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")},
        enable_logo=False,
        log_to_console=False,
        log_to_file=False,
        service={"backend": "cli"},
        components={"file_graph": {"default": {"backend": "local"}}},
        jobs={"health_check": {"backend": "base", "steps": [{"backend": "health_check_step"}]}},
    )
    async with app:
        alpha = app.context.workspaces["alpha"].get_component(ComponentEnum.FILE_GRAPH, "default")
        beta = app.context.workspaces["beta"].get_component(ComponentEnum.FILE_GRAPH, "default")
        await beta.upsert_nodes([FileNode(path="beta.md", st_mtime=1)])
        alpha_before = (await app.run_job("health_check", workspace="alpha")).metadata["health"]
        response = await app.run_job("health_check", workspace="beta")
        before = response.metadata["health"]
        assert before["components"]["file_graph"]["default"]["n_nodes"] == 1
        assert before["workspace"] == "beta" and before["healthy"] is True
        assert "beta" in response.answer
        assert before["components"]["file_graph"]["default"]["is_started"] is True
        await beta.close()
        response = await app.run_job("health_check", workspace="beta")
        assert response.success is True  # The diagnostic job succeeded; its health result is negative.
        after = response.metadata["health"]
        assert after["workspace"] == "beta" and after["healthy"] is False
        assert after["components"]["file_graph"]["default"]["is_started"] is False
        assert after["components"]["file_graph"]["default"]["n_nodes"] == 1
        assert alpha.is_started is True and beta.is_started is False
        assert (await app.run_job("health_check", workspace="alpha")).metadata["health"] == alpha_before
        assert (await app.run_job("health_check")).metadata["health"] == alpha_before


@pytest.mark.asyncio
@pytest.mark.parametrize("named", [False, True])
async def test_health_single_workspace_labels_default_and_preserves_payload(tmp_path, named):
    """String and one-entry mapping reports identify their default workspace."""
    context = ApplicationContext(workspace_dir={"named": str(tmp_path)} if named else str(tmp_path))
    alias = context.workspace.alias
    step = HealthCheckStep(app_context=context)
    response = await step()
    assert response.metadata["health"]["workspace"] == alias
    assert response.metadata["health"]["healthy"] is True
    assert set(response.metadata["health"]["components"]) == {
        "embedding_store",
        "file_graph",
        "file_store",
        "keyword_index",
    }


@pytest.mark.asyncio
async def test_reports_include_shared_and_selected_components_without_mutating_registries(tmp_path):
    """Shared and scoped names of the same type both remain visible; alpha never leaks."""
    context = ApplicationContext(workspace_dir={"alpha": str(tmp_path / "alpha"), "beta": str(tmp_path / "beta")})
    shared, alpha, beta = (BaseComponent() for _ in range(3))
    context.shared_components[ComponentEnum.FILE_GRAPH] = {"shared": shared}
    context.workspaces["alpha"].components[ComponentEnum.FILE_GRAPH] = {"alpha": alpha}
    context.workspaces["beta"].components[ComponentEnum.FILE_GRAPH] = {"beta": beta}
    await shared.start()
    await alpha.start()
    await beta.start()
    try:
        from reme.components.runtime_context import RuntimeContext

        step = HealthCheckStep(app_context=context)
        response = await step(RuntimeContext(workspace=context.workspaces["beta"]))
        assert set(response.metadata["health"]["components"]["file_graph"]) == {"shared", "beta"}
        memory = _collect_memory(context, context.workspaces["beta"])
        assert set(memory["components"]["file_graph"]) == {"shared", "beta"}
        await shared.close()
        response = await step(RuntimeContext(workspace=context.workspaces["beta"]))
        assert response.metadata["health"]["healthy"] is False
        assert set(context.shared_components[ComponentEnum.FILE_GRAPH]) == {"shared"}
        assert set(context.workspaces["alpha"].components[ComponentEnum.FILE_GRAPH]) == {"alpha"}
        assert set(context.workspaces["beta"].components[ComponentEnum.FILE_GRAPH]) == {"beta"}
    finally:
        await shared.close()
        await alpha.close()
        await beta.close()


def test_component_report_view_rejects_foreign_workspace(tmp_path):
    """A workspace object from another application cannot select this application's resources."""
    context = ApplicationContext(workspace_dir=str(tmp_path / "current"))
    foreign = ApplicationContext(workspace_dir=str(tmp_path / "foreign"))
    with pytest.raises(ValueError, match="not bound to this Application"):
        context.components_for(foreign.workspace)
