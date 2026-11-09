"""Opt-in real-model checks of cron, proactive and one-shot reminders using synthetic evidence."""

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path

from benchmark.proactive.cases import MEMORY, META, history
from benchmark.proactive.evaluate import grade
from benchmark.proactive.prompts import freeze
from benchmark.proactive.run import REPO, credentials, digest, sources
from benchmark.proactive.runtime import EFFORT, execute, save
from reme.config import resolve_app_config


def scenarios(now: datetime) -> dict:
    """Use a consistent current clock and explicit, fictional task facts."""
    stamp = now.isoformat()
    deadline = (now + timedelta(minutes=10)).strftime("%H:%M")
    task = {
        "id": "fixture-0",
        "cron": "*/5 * * * *",
        "enabled": True,
        "query": "看看 resource/fixtures/mail.md 里的项目邮件有没有需要我及时处理的新情况，只用本地资料，不联网。",
    }
    mail = (
        f"项目邮件的本地缓存。记录时间：{stamp}。\n"
        f"项目负责人刚来邮件：今天{deadline}前请主人确认演示讲稿用A版还是B版，逾期默认按A版安排。主人尚未回复。\n"
        "来源：https://mail.example.test/messages/demo-42\n"
    )
    common = {
        "now": stamp,
        "simulate_idle": True,
        "reminders": [],
        "memory": "# 个人记忆\n- 中文，简短直接。只分享有用的新结果；没新信息就安静，不问候。\n",
        "messages": history(("user", "帮我留意需要及时确认的项目邮件，有新情况简短告诉我。")),
        "initial_files": {
            f"{META}/proactive.json": json.dumps(
                {
                    "start_time": "0000",
                    "end_time": "0000",
                    "interval_minutes": 15,
                    "hints_enabled": False,
                },
            ),
        },
    }
    cases = {}
    for name in ("urgent", "routine", "duplicate", "changed"):
        case = deepcopy(common)
        case.update(
            role="cron",
            tasks=[task],
            resources={"resource/fixtures/mail.md": mail},
            expected_send=name in {"urgent", "changed"},
        )
        case["required"] = [deadline, "A", "B"]
        if name == "routine":
            case["resources"]["resource/fixtures/mail.md"] = (
                mail.replace(
                    f"今天{deadline}前",
                    f"{(now + timedelta(days=7)):%Y-%m-%d}前",
                )
                + "负责人明确说明不需要今天处理。\n"
            )
        if name in {"duplicate", "changed"}:
            earlier = deadline if name == "duplicate" else (now + timedelta(minutes=30)).strftime("%H:%M")
            case["reminders"] = [
                {
                    "status": "sent",
                    "content": f"主人，负责人请您今天{earlier}前确认演示讲稿A版还是B版，逾期默认A版。",
                },
            ]
        cases[f"cron_{name}"] = case
    for name, query, required in (
        (
            "due",
            "到约定时间了，提醒主人给绿植浇水。只完成本条提醒，不安排其他事项。",
            ["浇水"],
        ),
        (
            "result",
            "阅读 resource/fixtures/options.md，给我选方案的结论和理由，只用本地资料。",
            ["B", "离线"],
        ),
    ):
        case = deepcopy(common)
        case.update(
            role="temporary",
            query=query,
            tasks=[],
            expected_send=True,
            required=required,
            resources={},
        )
        case["messages"] = history(("user", query))
        if name == "result":
            case["resources"] = {
                "resource/fixtures/options.md": "我的硬要求：离线可用，预算500元以内。方案A：200元，必须联网。方案B：300元，离线可用。仅比较这两项。",
            }
        cases[f"temporary_{name}"] = case
    result = "对话读取的角色筛选、最新消息筛选和正文截断，新增37项本地测试全部通过。尚未验证浏览器界面。"
    for name in ("new", "known"):
        case = deepcopy(common)
        path = f"{META}/cron/fixture-0/{now:%Y-%m-%d}/{now:%H%M%S}.md"
        case.update(
            role="proactive",
            results_only=True,
            tasks=[{**task, "query": "读取本地对话读取验证报告，整理新增结果。"}],
            resources={"resource/fixtures/validation.md": result},
            expected_send=name == "new",
            required=["37", "本地"],
            messages=history(
                (
                    "user",
                    "这次帮我看一下对话读取的新验证结果，按本地资料简短告诉我；已经说过的就不重复，没有新结果就安静，也不用问候。不要联网。",
                ),
            ),
        )
        case["initial_files"][path] = (
            f"---\ntask_id: fixture-0\ncreated_at: '{stamp}'\nstatus: unread\n---\n{result}\n"
            "来源：resource/fixtures/validation.md\n"
        )
        if name == "known":
            case["reminders"] = [
                {
                    "status": "sent",
                    "content": "主人，对话读取新增37项本地测试全部通过，浏览器界面还没验证。",
                },
            ]
        cases[f"proactive_{name}"] = case
    for case in cases.values():
        for row in case["messages"]:
            row["created_at"] = stamp
    for name, evidence in (
        ("seen_page", "主人刚浏览了一个主动助手设计页面，没有新增发布、结果、风险或期限。"),
        ("seen_reply", "主人刚问过主动提醒的设计方法，助手已回答跟踪明确委托；没有其他新增内容。"),
        ("empty_history", "主人正在浏览项目页面，没有新增发布、结果、风险或期限。"),
    ):
        case = deepcopy(cases["proactive_known"])
        case["reminders"] = []
        case["resources"] = {"resource/fixtures/validation.md": evidence}
        case["tasks"][0]["query"] = "记录当前屏幕活动的新发现。"
        case["messages"] = [] if name == "empty_history" else history(("user", evidence))
        for row in case["messages"]:
            row["created_at"] = stamp
        path = next(path for path in case["initial_files"] if path.endswith(".md"))
        case["initial_files"][path] = case["initial_files"][path].replace(result, evidence)
        cases[f"proactive_{name}"] = case
    return cases


def checks(case: dict, record: dict) -> dict[str, bool]:
    """Check protocol, facts and message bounds; readability still requires reading actual messages."""
    sends = [
        job
        for job in record["jobs"]
        if job["name"] in {"send_chat_message", "send_chat_message_immediate"}
        and job["args"].get("content", "").strip()
    ]
    messages = [job["args"]["content"] for job in sends]
    good_jobs = all("error" not in job and job.get("response", {}).get("success", True) for job in record["jobs"])
    verdict = {
        "completed": record.get("completed", False),
        "tool_calls_succeeded": good_jobs,
        "expected_decision": len(sends) == int(case["expected_send"]),
        "memory_preserved": record.get("after", {}).get(MEMORY) == record.get("before", {}).get(MEMORY),
        "message_size": all(len(text) <= (4000 if case["role"] == "proactive" else 3980) for text in messages),
        "required_facts": not case["expected_send"]
        or bool(messages)
        and all(word in messages[0] for word in case["required"]),
        "fixture_delivery": len(record.get("delivered", [])) == int(case["expected_send"]),
    }
    if case["role"] == "proactive":
        verdict.update(grade("quality_case", {**record, "fixture": case})["checks"])
    if case["role"] == "cron":
        history_jobs = [job for job in record["jobs"] if job["name"] == "get_chat_messages"]
        verdict["message_phase_only"] = all(job["turn_index"] == 2 for job in sends + history_jobs)
        verdict["history_before_send"] = (
            not sends or bool(history_jobs) and record["jobs"].index(history_jobs[0]) < record["jobs"].index(sends[0])
        )
        verdict["no_unneeded_history"] = case["expected_send"] or bool(case["reminders"]) or not history_jobs
    return verdict


def report(root: Path, results: dict) -> None:
    """Keep the actual wording and failed checks reviewable without opening raw traces."""
    lines = [
        "# 提醒 Prompt 实测",
        "",
        "真实模型、合成资料、隔离 Main；模拟空闲后写入测试会话，未验证桌面展示。",
        "",
        "自动检查只覆盖流程、关键事实和消息边界；以下原文供人工判断是否自然、准确。",
        "",
    ]
    for name, result in results.items():
        failed = [key for key, value in result["checks"].items() if not value]
        lines += [
            f"## {name}",
            "",
            f"检查：{'通过' if not failed else '未通过：' + ', '.join(failed)}；{result['duration_s']}秒。",
            "",
        ]
        if result.get("error"):
            lines += [result["error"], ""]
        lines += [*result["messages"], ""] if result["messages"] else ["未发送。", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    save(root / "results.json", results)


async def run(root: Path, timeout: float, selected: str | None = None) -> None:
    """Execute fresh cases with production prompts and isolated native model sessions."""
    if root.exists() or root.is_relative_to(REPO):
        raise ValueError("Choose a fresh output directory outside the repository")
    cases = scenarios(datetime.now().astimezone().replace(microsecond=0))
    if selected:
        names = selected.split(",")
        if set(names) - cases.keys():
            raise ValueError("Unknown case name")
        cases = {name: cases[name] for name in names}
    auth = credentials()
    cfg = resolve_app_config(config="proactive", log_config=False)
    cfg["components"]["codex_app_server"]["default"].update(api_key=auth["LLM_API_KEY"], base_url=auth["LLM_BASE_URL"])
    bundle = freeze(REPO)["current"]
    model = cfg["components"]["agent_wrapper"]["default"]["model"]
    root.mkdir(parents=True)
    inputs = {"prompts": bundle, "cases": cases, "source_hashes": sources()}
    save(
        root / "manifest.json",
        {
            **inputs,
            "fingerprint": digest(inputs),
            "model": model,
            "effort": EFFORT,
            "delivery": "synthetic Main; simulated idle; no desktop",
        },
    )
    results, semaphore = {}, asyncio.Semaphore(2)

    async def one(name, case):
        async with semaphore:
            print(f"Starting {name}", flush=True)
            record = await execute(cfg, model, bundle, case, root / name, timeout)
            save(root / name / "record.json", record)
            messages = [
                job["args"]["content"]
                for job in record["jobs"]
                if job["name"] in {"send_chat_message", "send_chat_message_immediate"}
                and job["args"].get("content", "").strip()
            ]
            results[name] = {
                "checks": checks(case, record),
                "messages": messages,
                "duration_s": record["duration_s"],
                "error": record.get("error"),
            }
            report(root, results)
            print(
                json.dumps({"case": name, **results[name]}, ensure_ascii=False),
                flush=True,
            )

    await asyncio.gather(*(one(name, case) for name, case in cases.items()))
    if any(not all(result["checks"].values()) for result in results.values()):
        raise SystemExit(f"Reminder checks failed; see {root / 'report.md'}")


def main():
    """Parse the opt-in quality benchmark command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--cases", help="Comma-separated scenario names")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    asyncio.run(run(args.root.expanduser().resolve(), args.timeout, args.cases))


if __name__ == "__main__":
    main()
