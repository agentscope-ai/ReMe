"""Temporary package skills for generic Wrapper publication contracts."""


def install_builtin(monkeypatch, tmp_path, name):
    """Keep Wrapper builtin-source tests independent of proactive YAML templates."""
    root = tmp_path / "builtin_package"
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test builtin\n---\nRead the source.\n")
    monkeypatch.setattr("reme.components.agent_wrapper.skills.BUILTIN_SKILLS_ROOT", root)
    return directory
