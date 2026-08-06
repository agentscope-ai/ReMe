"""Select configured ETFs that are directly related to today's news."""

from __future__ import annotations

import json
from typing import Any

from ....components import R
from ....schema import AutoFinEtfsOutput
from ._base import AutoFinStep, _write


@R.register("auto_fin_topic_step")
class AutoFinTopicStep(AutoFinStep):
    """Call the first tool-free Agent with complete current-news context."""

    @staticmethod
    def _normalize(
        output: AutoFinEtfsOutput,
        news: list[dict[str, str]],
        names: dict[str, str],
        limit: int,
    ):
        news_ids = {row["news_id"] for row in news}
        selected: dict[str, dict[str, Any]] = {}
        for item in output.etfs:
            code = item.etf_code.strip().upper()
            if code not in names:
                continue
            target = selected.setdefault(code, {"etf_code": code, "etf_name": names[code], "events": []})
            seen = {event["news_id"] for event in target["events"]}
            for event in item.events:
                news_id, reason = event.news_id.strip(), event.reason.strip()
                if news_id in news_ids and news_id not in seen and reason and len(target["events"]) < limit:
                    target["events"].append({"news_id": news_id, "reason": reason})
                    seen.add(news_id)
        return AutoFinEtfsOutput.model_validate({"etfs": [item for item in selected.values() if item["events"]]})

    async def execute(self):
        assert self.context is not None
        if self.context.get("auto_fin_skipped"):
            return self.context.response
        from .data import AutoFinDataStep  # Avoid a module import cycle.

        day = str(self._required("auto_fin_date"))
        news_path = self.workspace_path / str(self.config_value("daily_dir")) / day / "auto_fin_news.md"
        news = AutoFinDataStep.read_news(news_path)
        names = dict(self._required("auto_fin_etf_names"))
        output, output_path = await self._reply(
            "topic_user",
            "auto_fin_topic",
            AutoFinEtfsOutput,
            news=json.dumps(news, ensure_ascii=False),
            etfs=json.dumps(
                [{"etf_code": code, "etf_name": name} for code, name in names.items()],
                ensure_ascii=False,
            ),
        )
        normalized = self._normalize(output, news, names, int(self._value("current_news_limit_per_etf", 10)))
        if normalized != output:
            _write(output_path, normalized.model_dump_json() + "\n")
        self.context["auto_fin_news"] = news
        self.context["auto_fin_etfs"] = normalized.model_dump(mode="json")["etfs"]
        self.context.response.metadata.update(
            {"news_count": len(news), "etf_count": len(normalized.etfs)},
        )
        return self.context.response
