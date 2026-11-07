"""Regression scenarios for useful reminders and silent empty decisions in the current workflow."""

import argparse
import asyncio
from copy import deepcopy
from pathlib import Path

from benchmark.proactive.cases import CASES, MEMORY, history
from benchmark.proactive.evaluate import grade
from benchmark.proactive.prompts import freeze
from benchmark.proactive.run import REPO, credentials
from benchmark.proactive.runtime import execute, save
from reme.config import resolve_app_config

SEEN_PAGE = "主人正在阅读 ReMe 开源项目页面，讨论其主动式助手设计；没有新的发布、结果、风险或期限。"
SEEN_REPLY = "主人刚问过主动提醒产品的设计方法，助手已经回答先跟踪明确委托的任务；没有其他新增内容。"


def scenarios() -> dict:
    cases = {}
    for name, evidence in (("seen_page", SEEN_PAGE), ("seen_reply", SEEN_REPLY), ("empty_history", SEEN_PAGE)):
        cases[f"proactive_{name}"] = {
            "role": "proactive", "expected_send": False,
            "messages": [] if name == "empty_history" else history(("user", evidence), ("assistant", "主人，了解。")),
            "resources": {"resource/fixtures/screen.md": evidence}, "reminders": [],
        }
    cases["proactive_awaited_result"] = deepcopy(CASES["proactive_timely"])
    return cases


def checks(case: dict, record: dict, expected_send: bool) -> dict[str, bool]:
    fixture = {**case, "expected_send": expected_send}
    verdict = grade("quality_case", {**record, "fixture": fixture})["checks"]
    verdict["memory_preserved"] = record.get("after", {}).get(MEMORY) == record.get("before", {}).get(MEMORY)
    return verdict


async def run(root: Path, timeout: float) -> None:
    if root.exists() or root.is_relative_to(REPO):
        raise ValueError("Choose a fresh output directory outside the repository")
    auth = credentials()
    cfg = resolve_app_config(config="proactive", log_config=False)
    cfg["components"]["codex_app_server"]["default"].update(api_key=auth["LLM_API_KEY"], base_url=auth["LLM_BASE_URL"])
    bundle = freeze(REPO)["current"]
    model = cfg["components"]["agent_wrapper"]["default"]["model"]
    root.mkdir(parents=True)
    results = {}
    for name, case in scenarios().items():
        record = await execute(cfg, model, bundle, case, root / name, timeout)
        save(root / name / "record.json", record)
        results[name] = checks(case, record, case["expected_send"])
    save(root / "results.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    asyncio.run(run(args.root.expanduser().resolve(), args.timeout))


if __name__ == "__main__":
    main()
