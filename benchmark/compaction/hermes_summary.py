"""Run the real Hermes summary generator with a small-window adapter and pinned Qwen transport."""

# Private generator hooks are intentionally pinned to the Hermes revision recorded in the corpus.
# pylint: disable=protected-access

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

from dotenv import dotenv_values
from openai import OpenAI

from benchmark.compaction.cases import write


def run(args):
    """Isolate state, apply host budgets to the native template and retain wire bodies."""
    root = args.output.resolve()
    root.mkdir(parents=True)
    home = root / "hermes_home"
    home.mkdir()
    os.environ["HERMES_HOME"] = str(home)
    (home / "config.yaml").write_text("model:\n  provider: custom\n  context_length: 131072\n")
    values = dotenv_values(args.env_file)
    sys.path.insert(0, str(args.hermes.resolve()))
    import agent.context_compressor as cc

    source = json.loads(args.source.read_text())
    history = source["messages"]
    # The source is the already validated public history, with the final report excluded.
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    allowance = json.loads((args.source.parent / "budget.json").read_text())["allowance"]
    records = []
    index = 0
    attempt = 0

    class ForcedSummary(cc.ContextCompressor):
        """Model the small-window budget without altering the installed Hermes checkout."""

        target = allowance

        def _compute_summary_budget(self, turns_to_summarize):  # pylint: disable=unused-argument
            return self.target

        def _build_summary_prompt(
            self,
            content_to_summarize,
            summary_budget,
            focus_topic,
            memory_context,
            has_user_turn,
        ):
            prompt = super()._build_summary_prompt(
                content_to_summarize,
                summary_budget,
                focus_topic,
                memory_context,
                has_user_turn,
            )
            words = max(18, int(self.target * 0.35))
            return prompt + (
                f"\n\nSMALL-WINDOW SIMULATION: Write at most {words} English words TOTAL. "
                "This total budget takes precedence over the detailed template above. "
                "Keep the Historical Task Snapshot section and its required no-user sentinel. "
                "Keep a Critical Context section with the most useful exact facts. "
                "Omit every other section, empty entry, repeated task and prose when necessary. "
                "No preamble. Do not include the context-compaction control prefix."
            )

    def transport(**kwargs):
        request = {
            "model": args.model,
            "messages": kwargs["messages"],
            "temperature": 0,
            "reasoning_effort": "none",
            "extra_body": {"enable_thinking": False},
        }
        request_file = root / f"c{index}.a{attempt}.summary.request.json"
        response_file = root / f"c{index}.a{attempt}.summary.response.json"
        write(request_file, request)
        response = client.chat.completions.create(**request)
        write(response_file, response.model_dump(mode="json"))
        kwargs.get("route_info", {}).update(provider="custom", model=args.model)
        records.append(
            {
                "generation": index,
                "usage": response.usage.model_dump(),
                "finish_reason": response.choices[0].finish_reason,
            },
        )
        return response

    client = OpenAI(api_key=values["LLM_API_KEY"], base_url=values["LLM_BASE_URL"], timeout=90, max_retries=1)
    original = cc.call_llm
    cc.call_llm = transport
    summaries = []
    try:
        for index in range(args.compressions):
            accepted = None
            for attempt in range(3):
                compressor = ForcedSummary(
                    model=args.model,
                    base_url=values["LLM_BASE_URL"],
                    api_key=values["LLM_API_KEY"],
                    provider="custom",
                    api_mode="chat_completions",
                    config_context_length=131072,
                    quiet_mode=True,
                    tail_mode="legacy",
                )
                compressor.target = max(12, int(allowance * 0.8**attempt))
                summary = compressor._generate_summary(
                    deepcopy(history[2:]),
                    focus_topic=history[1]["content"],
                )  # pylint: disable=protected-access
                body = compressor._strip_summary_prefix(summary) if summary else ""  # pylint: disable=protected-access
                if body and len(tokenizer.encode(body, add_special_tokens=False).ids) <= allowance:
                    accepted = body
                    break
            summaries.append(
                {
                    "repeat": index,
                    "summary": accepted,
                    "native_budget": allowance,
                    "summary_tokens": len(tokenizer.encode(accepted or "", add_special_tokens=False).ids),
                    "valid": accepted is not None,
                    "error": None if accepted else "summary_budget_exceeded",
                },
            )
            write(root / "summaries.json", summaries)
            print(
                args.source.parent.name,
                "native small-window summary",
                index,
                "budget",
                allowance,
                "OK" if accepted else "FAILED",
                flush=True,
            )
        write(root / "usage.json", records)
        for file in root.rglob("*"):
            if file.is_file():
                assert values["LLM_API_KEY"].encode() not in file.read_bytes()
        write(
            root / "validation.json",
            {
                "no_api_keys": True,
                "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                "generator": (
                    "Hermes legacy generator with explicit small-window prompt/budget adapter; body-only carrier"
                ),
            },
        )
    finally:
        cc.call_llm = original
        client.close()


def main():
    """Internal subprocess entry point; the orchestrator owns the live-call opt-in."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hermes", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--compressions", type=int, default=3)
    parser.add_argument("--model", default="qwen3.8-max")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
