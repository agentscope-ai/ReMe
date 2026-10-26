"""Send a workspace Markdown file to DingTalk group conversations."""

import asyncio
import json

import aiofiles
import frontmatter
import httpx

from ....components import R
from ...base_step import BaseStep
from ...file_io._path import gate_md, resolve_path

_GROUP_SEND_URL = "https://api.dingtalk.com/v1.0/robot/groupMessages/send"


def _conversation_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@R.register("dingtalk_markdown_send_step")
class DingTalkMarkdownSendStep(BaseStep):
    """Send one Markdown document serially to configured DingTalk groups."""

    def __init__(
        self,
        app_key: str = "",
        app_secret: str = "",
        robot_code: str = "",
        conversation_ids: str = "",
        title: str = "",
        timeout: float = 15.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.app_key = app_key
        self.app_secret = app_secret
        self.robot_code = robot_code
        self.conversation_ids = _conversation_ids(conversation_ids)
        self.title = title
        self.timeout = timeout

    async def execute(self):
        assert self.context is not None
        recipients = self.conversation_ids
        self.context.response.metadata["dingtalk_configured_count"] = len(recipients)
        self.context.response.metadata["dingtalk_sent_count"] = 0
        if not recipients:
            self.logger.info(f"[{self.name}] skipped DingTalk Markdown delivery: no conversation IDs")
            return self.context.response
        if not all((self.app_key, self.app_secret, self.robot_code)):
            raise RuntimeError("DingTalk Markdown delivery requires app_key, app_secret, and robot_code")

        raw_path = str(self.context.get("markdown_path") or "")
        if not raw_path:
            if self.context.response.metadata.get("skipped"):
                self.logger.info(f"[{self.name}] skipped DingTalk Markdown delivery: no markdown path")
                return self.context.response
            raise RuntimeError("DingTalk Markdown delivery requires markdown_path")

        target, error = resolve_path(self.workspace_path, raw_path)
        if error:
            raise ValueError(f"Invalid DingTalk Markdown path: {error}")
        assert target is not None
        target, is_markdown = gate_md(target)
        if not is_markdown:
            raise ValueError("DingTalk Markdown delivery requires a .md file")
        if not target.is_file():
            raise FileNotFoundError(f"DingTalk Markdown file does not exist: {raw_path}")

        async with aiofiles.open(target, encoding="utf-8") as stream:
            document = frontmatter.loads(await stream.read())
        markdown = document.content.strip()
        if not markdown:
            raise ValueError(f"DingTalk Markdown file is empty: {raw_path}")

        import dingtalk_stream  # pylint: disable=import-outside-toplevel

        title = self.title or str(document.metadata.get("name") or target.stem)
        token_client = dingtalk_stream.DingTalkStreamClient(
            dingtalk_stream.Credential(self.app_key, self.app_secret),
        )
        access_token = await asyncio.to_thread(token_client.get_access_token)
        if not access_token:
            raise RuntimeError("Failed to obtain DingTalk access token")

        self.logger.info(
            f"[{self.name}] sending DingTalk Markdown path={raw_path} recipients={len(recipients)} "
            f"chars={len(markdown)} timeout={self.timeout:.1f}s",
        )
        failures: list[str] = []
        headers = {"x-acs-dingtalk-access-token": access_token, "User-Agent": "ReMe DingTalk notifier"}
        transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0")
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers, transport=transport) as client:
            for index, conversation_id in enumerate(recipients, start=1):
                payload = {
                    "robotCode": self.robot_code,
                    "openConversationId": conversation_id,
                    "msgKey": "sampleMarkdown",
                    "msgParam": json.dumps({"title": title, "text": markdown}, ensure_ascii=False),
                }
                try:
                    response = await client.post(_GROUP_SEND_URL, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    if not isinstance(result, dict) or not result.get("processQueryKey"):
                        raise ValueError("missing processQueryKey")
                except (httpx.HTTPError, ValueError) as exc:
                    failures.append(f"recipient {index}: {type(exc).__name__}")
                    self.logger.warning(
                        f"[{self.name}] DingTalk delivery failed recipient={index}/{len(recipients)} "
                        f"error_type={type(exc).__name__}",
                    )
                    continue
                self.context.response.metadata["dingtalk_sent_count"] += 1
                self.logger.info(f"[{self.name}] delivered DingTalk Markdown recipient={index}/{len(recipients)}")

        sent_count = self.context.response.metadata["dingtalk_sent_count"]
        if failures:
            self.context.response.metadata["dingtalk_delivery_errors"] = failures
            raise RuntimeError(f"DingTalk Markdown delivery failed for {len(failures)} of {len(recipients)} recipients")
        self.logger.info(f"[{self.name}] DingTalk Markdown delivery complete sent={sent_count} total={len(recipients)}")
        return self.context.response
