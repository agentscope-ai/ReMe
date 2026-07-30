#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
钉钉消息发送工具
支持多机器人管理、企业内部机器人和 Webhook 自定义机器人
支持机器人状态追踪和最近使用记录
"""

import argparse
import hashlib
import hmac
import base64
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

# ==================== 路径常量 ====================


def _get_config_dir() -> Path:
    """获取跨平台的配置目录
    - Windows: %APPDATA%/dingtalk/
    - macOS/Linux: ~/.config/dingtalk/
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "dingtalk"


CONFIG_DIR = _get_config_dir()
CONFIG_PATH = CONFIG_DIR / "config.json"
STATE_PATH = CONFIG_DIR / "state.json"


# ==================== 配置与状态管理 ====================


def _load_json(path: Path) -> dict:
    """从文件加载 JSON，不存在则返回空 dict"""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_json(path: Path, data: dict):
    """保存 JSON 到文件，自动创建父目录"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_config() -> dict:
    """加载配置文件"""
    return _load_json(CONFIG_PATH)


def save_config(config: dict):
    """保存配置文件"""
    _save_json(CONFIG_PATH, config)


def load_state() -> dict:
    """加载状态文件"""
    if not STATE_PATH.exists():
        return {"robots": {}}
    with open(STATE_PATH, "r") as f:
        return json.load(f)


def save_state(state: dict):
    """保存状态文件"""
    _save_json(STATE_PATH, state)


def ensure_config() -> dict:
    """加载配置"""
    config = load_config()
    if not config:
        return {"default_robot": None, "robots": []}
    return config


def find_robot(
    config: dict, name: Optional[str] = None, robot_type: Optional[str] = None
) -> Optional[dict]:
    """
    查找机器人，优先级：
    1. 指定名称
    2. default_robot
    3. last_used 最近的（匹配类型）
    4. 第一个匹配类型的
    """
    robots = config.get("robots", [])
    if not robots:
        return None

    state = load_state()
    robot_states = state.get("robots", {})

    # 1. 按名称查找
    if name:
        for r in robots:
            if r["name"] == name:
                if robot_type and r["type"] != robot_type:
                    print(
                        f"警告：机器人 '{name}' 类型为 {r['type']}，但当前命令需要 {robot_type} 类型"
                    )
                return r
        print(f"错误：找不到名为 '{name}' 的机器人")
        print(f"可用机器人：{', '.join(r['name'] for r in robots)}")
        sys.exit(1)

    # 按类型过滤（只保留启用的）
    candidates = []
    for r in robots:
        if robot_type and r["type"] != robot_type:
            continue
        rs = robot_states.get(r["name"], {})
        if rs.get("enabled", True):
            candidates.append(r)

    if not candidates:
        return None

    # 2. default_robot
    default_name = config.get("default_robot")
    if default_name:
        for r in candidates:
            if r["name"] == default_name:
                return r

    # 3. last_used 最近的
    def get_last_used(r):
        rs = robot_states.get(r["name"], {})
        return rs.get("last_used", "")

    candidates.sort(key=get_last_used, reverse=True)
    if get_last_used(candidates[0]):
        return candidates[0]

    # 4. 第一个
    return candidates[0]


def update_robot_state(
    name: str, success: bool = True, msg_summary: Optional[str] = None
):
    """更新机器人使用状态，记录消息摘要"""
    state = load_state()
    if "robots" not in state:
        state["robots"] = {}

    if name not in state["robots"]:
        state["robots"][name] = {
            "enabled": True,
            "last_used": None,
            "use_count": 0,
            "last_status": None,
            "recent_messages": [],
        }

    rs = state["robots"][name]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rs["last_used"] = now
    rs["use_count"] = rs.get("use_count", 0) + 1
    rs["last_status"] = "success" if success else "failed"

    # 记录最近消息摘要（保留最近10条）
    if msg_summary:
        recent = rs.get("recent_messages", [])
        recent.append(
            {
                "time": now,
                "summary": msg_summary[:80],
                "status": "ok" if success else "fail",
            }
        )
        rs["recent_messages"] = recent[-10:]

    save_state(state)


# ==================== 数据类 ====================


@dataclass
class DingTalkConfig:
    """钉钉企业内部机器人配置"""

    app_key: str
    app_secret: str
    robot_code: Optional[str] = None
    agent_id: Optional[str] = None

    BASE_URL = "https://api.dingtalk.com"
    OLD_BASE_URL = "https://oapi.dingtalk.com"

    def __post_init__(self):
        if not self.app_key or not self.app_secret:
            raise ValueError("app_key 和 app_secret 不能为空")


@dataclass
class DingTalkWebhookConfig:
    """钉钉 Webhook 自定义机器人配置"""

    webhook_token: str
    secret: Optional[str] = None

    WEBHOOK_BASE = "https://oapi.dingtalk.com/robot/send"

    def __post_init__(self):
        if not self.webhook_token:
            raise ValueError("webhook_token 不能为空")

    @property
    def webhook_url(self) -> str:
        """根据 token 生成完整 URL"""
        if self.webhook_token.startswith("http"):
            return self.webhook_token
        return f"{self.WEBHOOK_BASE}?access_token={self.webhook_token}"


# ==================== Webhook 发送器 ====================


class DingTalkWebhookSender:
    """钉钉 Webhook 自定义机器人消息发送器"""

    def __init__(self, config: Union[DingTalkWebhookConfig, Dict]):
        if isinstance(config, dict):
            self.config = DingTalkWebhookConfig(**config)
        else:
            self.config = config
        try:
            import requests

            self.session = requests.Session()
        except ImportError:
            raise ImportError("请安装 requests: pip install requests")

    def _build_url(self) -> str:
        url = self.config.webhook_url
        if self.config.secret:
            timestamp = str(round(time.time() * 1000))
            string_to_sign = f"{timestamp}\n{self.config.secret}"
            hmac_code = hmac.new(
                self.config.secret.encode("utf-8"),
                string_to_sign.encode("utf-8"),
                digestmod=hashlib.sha256,
            ).digest()
            sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}timestamp={timestamp}&sign={sign}"
        return url

    def _send(self, payload: Dict) -> Dict:
        url = self._build_url()
        headers = {"Content-Type": "application/json; charset=utf-8"}
        try:
            response = self.session.post(url, headers=headers, json=payload)
            if not response.ok:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise Exception(f"Webhook HTTP {response.status_code}: {detail}")
            return response.json()
        except Exception as e:
            if "HTTP " in str(e):
                raise
            raise Exception(f"Webhook 请求失败: {str(e)}")

    @staticmethod
    def _process_markdown_text(text: str) -> str:
        return DingTalkMessenger._process_markdown_text(text)

    def send_text(
        self,
        content: str,
        at_mobiles: Optional[List[str]] = None,
        at_user_ids: Optional[List[str]] = None,
        is_at_all: bool = False,
    ) -> Dict:
        content = content.replace("\\n", "\n")
        payload = {
            "msgtype": "text",
            "text": {"content": content},
            "at": {"isAtAll": is_at_all},
        }
        if at_mobiles:
            payload["at"]["atMobiles"] = at_mobiles
        if at_user_ids:
            payload["at"]["atUserIds"] = at_user_ids
        return self._send(payload)

    def send_markdown(
        self,
        title: str,
        text: str,
        at_mobiles: Optional[List[str]] = None,
        at_user_ids: Optional[List[str]] = None,
        is_at_all: bool = False,
    ) -> Dict:
        processed_text = self._process_markdown_text(text)
        payload = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": processed_text},
            "at": {"isAtAll": is_at_all},
        }
        if at_mobiles:
            payload["at"]["atMobiles"] = at_mobiles
        if at_user_ids:
            payload["at"]["atUserIds"] = at_user_ids
        return self._send(payload)

    def send_link(
        self, title: str, text: str, message_url: str, pic_url: Optional[str] = None
    ) -> Dict:
        text = text.replace("\\n", "\n")
        payload = {
            "msgtype": "link",
            "link": {"title": title, "text": text, "messageUrl": message_url},
        }
        if pic_url:
            payload["link"]["picUrl"] = pic_url
        return self._send(payload)

    def send_action_card(
        self,
        title: str,
        text: str,
        single_title: str,
        single_url: str,
        btn_orientation: str = "0",
    ) -> Dict:
        text = self._process_markdown_text(text)
        payload = {
            "msgtype": "actionCard",
            "actionCard": {
                "title": title,
                "text": text,
                "btnOrientation": btn_orientation,
                "singleTitle": single_title,
                "singleURL": single_url,
            },
        }
        return self._send(payload)

    def send_action_card_multi(
        self, title: str, text: str, buttons: List[Dict], btn_orientation: str = "0"
    ) -> Dict:
        text = self._process_markdown_text(text)
        payload = {
            "msgtype": "actionCard",
            "actionCard": {
                "title": title,
                "text": text,
                "btnOrientation": btn_orientation,
                "btns": buttons,
            },
        }
        return self._send(payload)

    def send_feed_card(self, links: List[Dict]) -> Dict:
        payload = {"msgtype": "feedCard", "feedCard": {"links": links}}
        return self._send(payload)


# ==================== 企业内部机器人发送器 ====================


class DingTalkMessenger:
    """钉钉企业内部机器人消息发送器"""

    MSG_TYPES = {
        "text": "sampleText",
        "image": "sampleImage",
        "voice": "sampleVoice",
        "file": "sampleFile",
        "link": "sampleLink",
        "markdown": "sampleMarkdown",
        "action_card": "sampleActionCard",
        "action_card_multi": "sampleMultiActionCard",
        "group_text": "text",
        "group_image": "image",
        "group_link": "link",
        "group_markdown": "markdown",
        "group_action_card": "actionCard",
        "group_feed_card": "feedCard",
    }

    def __init__(self, config: Union[DingTalkConfig, Dict]):
        if isinstance(config, dict):
            self.config = DingTalkConfig(**config)
        else:
            self.config = config
        self._access_token = None
        self._token_expire_time = 0
        try:
            import requests

            self.session = requests.Session()
        except ImportError:
            raise ImportError("请安装 requests: pip install requests")

    def _get_access_token(self) -> str:
        current_time = time.time()
        if self._access_token and current_time < self._token_expire_time - 300:
            return self._access_token
        url = f"{self.config.OLD_BASE_URL}/gettoken"
        params = {"appkey": self.config.app_key, "appsecret": self.config.app_secret}
        try:
            response = self.session.get(url, params=params)
            if not response.ok:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise Exception(
                    f"获取access_token失败 HTTP {response.status_code}: {detail}"
                )
            data = response.json()
            if data.get("errcode") != 0:
                raise Exception(f"获取access_token失败: {data}")
            self._access_token = data["access_token"]
            self._token_expire_time = current_time + data.get("expires_in", 7200)
            return self._access_token
        except Exception as e:
            if "access_token" in str(e):
                raise
            raise Exception(f"请求access_token失败: {str(e)}")

    def _upload_media(self, file_path: str, media_type: str = "file") -> str:
        url = f"{self.config.OLD_BASE_URL}/media/upload"
        params = {"access_token": self._get_access_token()}
        with open(file_path, "rb") as f:
            files = {"media": f}
            data = {"type": media_type}
            try:
                response = self.session.post(url, params=params, files=files, data=data)
                if not response.ok:
                    try:
                        detail = response.json()
                    except Exception:
                        detail = response.text
                    raise Exception(
                        f"上传文件失败 HTTP {response.status_code}: {detail}"
                    )
                result = response.json()
                if result.get("errcode") != 0:
                    raise Exception(f"上传文件失败: {result}")
                return result["media_id"]
            except Exception as e:
                if "上传文件" in str(e):
                    raise
                raise Exception(f"上传文件请求失败: {str(e)}")

    def _make_request(self, method: str, url: str, **kwargs) -> Dict:
        try:
            response = self.session.request(method, url, **kwargs)
            if not response.ok:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise Exception(f"HTTP {response.status_code}: {detail}")
            return response.json()
        except Exception as e:
            if "HTTP " in str(e):
                raise
            raise Exception(f"HTTP请求失败: {str(e)}")

    # ---- 单聊 ----

    def send_o2o_message(
        self, user_ids: List[str], msg_key: str, msg_param: Dict
    ) -> Dict:
        if not user_ids:
            raise ValueError("user_ids不能为空")
        if len(user_ids) > 100:
            raise ValueError("单次最多发送给100个用户")
        if not self.config.robot_code:
            raise ValueError("发送单聊消息需要配置robot_code")
        url = f"{self.config.BASE_URL}/v1.0/robot/oToMessages/batchSend"
        headers = {
            "x-acs-dingtalk-access-token": self._get_access_token(),
            "Content-Type": "application/json",
        }
        payload = {
            "robotCode": self.config.robot_code,
            "userIds": user_ids,
            "msgKey": msg_key,
            "msgParam": json.dumps(msg_param, ensure_ascii=False),
        }
        return self._make_request("POST", url, headers=headers, json=payload)

    def send_o2o_text(
        self, user_ids: List[str], content: str, at_users: Optional[List[str]] = None
    ) -> Dict:
        content = content.replace("\\n", "\n")
        msg_param = {"content": content}
        if at_users:
            msg_param["atUserIds"] = at_users
        return self.send_o2o_message(user_ids, self.MSG_TYPES["text"], msg_param)

    @staticmethod
    def _process_markdown_text(text: str) -> str:
        import re

        processed = text.replace("\\n", "\n")
        processed = re.sub(r"^\s*[-*_]{3,}\s*$", "", processed, flags=re.MULTILINE)
        processed = re.sub(r"\n{3,}", "\n\n", processed)
        return processed.strip()

    def send_o2o_markdown(self, user_ids: List[str], title: str, text: str) -> Dict:
        processed_text = self._process_markdown_text(text)
        msg_param = {"title": title, "text": processed_text}
        return self.send_o2o_message(user_ids, self.MSG_TYPES["markdown"], msg_param)

    def send_o2o_link(
        self,
        user_ids: List[str],
        title: str,
        text: str,
        message_url: str,
        pic_url: Optional[str] = None,
    ) -> Dict:
        msg_param = {"title": title, "text": text, "messageUrl": message_url}
        if pic_url:
            msg_param["picUrl"] = pic_url
        return self.send_o2o_message(user_ids, self.MSG_TYPES["link"], msg_param)

    def send_o2o_image(
        self, user_ids: List[str], media_id: str, caption: Optional[str] = None
    ) -> Dict:
        msg_param = {"mediaId": media_id}
        if caption:
            msg_param["caption"] = caption
        return self.send_o2o_message(user_ids, self.MSG_TYPES["image"], msg_param)

    def send_o2o_file(
        self,
        user_ids: List[str],
        media_id: str,
        file_name: str,
        file_size: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> Dict:
        msg_param = {"mediaId": media_id, "fileName": file_name}
        if file_size:
            msg_param["fileSize"] = file_size
        if file_type:
            msg_param["fileType"] = file_type
        return self.send_o2o_message(user_ids, self.MSG_TYPES["file"], msg_param)

    def send_o2o_voice(
        self,
        user_ids: List[str],
        media_id: str,
        duration: str,
        file_size: Optional[str] = None,
    ) -> Dict:
        msg_param = {"mediaId": media_id, "duration": duration}
        if file_size:
            msg_param["fileSize"] = file_size
        return self.send_o2o_message(user_ids, self.MSG_TYPES["voice"], msg_param)

    def send_o2o_action_card(
        self,
        user_ids: List[str],
        title: str,
        markdown: str,
        single_title: str,
        single_url: str,
    ) -> Dict:
        msg_param = {
            "title": title,
            "markdown": markdown,
            "singleTitle": single_title,
            "singleUrl": single_url,
        }
        return self.send_o2o_message(user_ids, self.MSG_TYPES["action_card"], msg_param)

    def send_o2o_action_card_multi(
        self,
        user_ids: List[str],
        title: str,
        markdown: str,
        buttons: List[Dict],
        btn_orientation: str = "0",
    ) -> Dict:
        msg_param = {
            "title": title,
            "markdown": markdown,
            "btnOrientation": btn_orientation,
            "btns": buttons,
        }
        return self.send_o2o_message(
            user_ids, self.MSG_TYPES["action_card_multi"], msg_param
        )

    # ---- 群聊 ----

    def send_group_message(
        self, open_conversation_id: str, msg_type: str, content: Dict
    ) -> Dict:
        if not open_conversation_id:
            raise ValueError("open_conversation_id不能为空")
        if not self.config.robot_code:
            raise ValueError("发送群聊消息需要配置robot_code")
        url = f"{self.config.BASE_URL}/v1.0/robot/groupMessages/send"
        headers = {
            "x-acs-dingtalk-access-token": self._get_access_token(),
            "Content-Type": "application/json",
        }
        payload = {
            "robotCode": self.config.robot_code,
            "openConversationId": open_conversation_id,
            "msgKey": msg_type,
            "msgParam": json.dumps(content, ensure_ascii=False),
        }
        return self._make_request("POST", url, headers=headers, json=payload)

    def send_group_text(
        self,
        open_conversation_id: str,
        content: str,
        at_mobiles: Optional[List[str]] = None,
        at_user_ids: Optional[List[str]] = None,
        is_at_all: bool = False,
    ) -> Dict:
        msg_param = {"content": content, "at": {"isAtAll": is_at_all}}
        if at_mobiles:
            msg_param["at"]["atMobiles"] = at_mobiles
        if at_user_ids:
            msg_param["at"]["atUserIds"] = at_user_ids
        return self.send_group_message(open_conversation_id, "sampleText", msg_param)

    def send_group_markdown(
        self,
        open_conversation_id: str,
        title: str,
        text: str,
        at_mobiles: Optional[List[str]] = None,
        at_user_ids: Optional[List[str]] = None,
        is_at_all: bool = False,
    ) -> Dict:
        msg_param = {"title": title, "text": self._process_markdown_text(text)}
        if at_mobiles or at_user_ids or is_at_all:
            msg_param["at"] = {"isAtAll": is_at_all}
            if at_mobiles:
                msg_param["at"]["atMobiles"] = at_mobiles
            if at_user_ids:
                msg_param["at"]["atUserIds"] = at_user_ids
        return self.send_group_message(
            open_conversation_id, "sampleMarkdown", msg_param
        )

    def send_group_link(
        self,
        open_conversation_id: str,
        title: str,
        text: str,
        message_url: str,
        pic_url: Optional[str] = None,
    ) -> Dict:
        msg_param = {"title": title, "text": text, "messageUrl": message_url}
        if pic_url:
            msg_param["picUrl"] = pic_url
        return self.send_group_message(open_conversation_id, "sampleLink", msg_param)

    def send_group_action_card(
        self,
        open_conversation_id: str,
        title: str,
        markdown: str,
        single_title: str,
        single_url: str,
    ) -> Dict:
        msg_param = {
            "title": title,
            "markdown": markdown,
            "singleTitle": single_title,
            "singleUrl": single_url,
        }
        return self.send_group_message(
            open_conversation_id, "sampleActionCard", msg_param
        )

    def send_group_action_card_multi(
        self,
        open_conversation_id: str,
        title: str,
        markdown: str,
        buttons: List[Dict],
        btn_orientation: str = "0",
    ) -> Dict:
        msg_param = {
            "title": title,
            "markdown": markdown,
            "btnOrientation": btn_orientation,
            "btns": buttons,
        }
        return self.send_group_message(
            open_conversation_id, "sampleMultiActionCard", msg_param
        )

    def send_group_feed_card(
        self, open_conversation_id: str, links: List[Dict]
    ) -> Dict:
        return self.send_group_message(
            open_conversation_id, "feedCard", {"links": links}
        )

    # ---- 文件上传 ----

    def upload_and_send_o2o_file(
        self, user_ids: List[str], file_path: str, file_name: Optional[str] = None
    ) -> Dict:
        import os

        media_id = self._upload_media(file_path, "file")
        if not file_name:
            file_name = os.path.basename(file_path)
        file_size = str(os.path.getsize(file_path))
        return self.send_o2o_file(user_ids, media_id, file_name, file_size)

    def upload_and_send_o2o_image(
        self, user_ids: List[str], file_path: str, caption: Optional[str] = None
    ) -> Dict:
        media_id = self._upload_media(file_path, "image")
        return self.send_o2o_image(user_ids, media_id, caption)


# ==================== 机器人解析与创建 ====================


def resolve_robot(args, robot_type: str) -> dict:
    """
    根据 --robot 参数和配置文件解析出目标机器人配置。
    当无任何机器人可用时，提示用户配置。

    Args:
        args: CLI 参数
        robot_type: "app" 或 "webhook"

    Returns:
        机器人配置 dict
    """
    config = ensure_config()
    robot_name = getattr(args, "robot", None)

    # 如果通过命令行传入了完整凭证，直接使用（不走机器人管理）
    if robot_type == "webhook":
        token = getattr(args, "webhook_token", None)
        if token:
            return {
                "name": "__cli__",
                "type": "webhook",
                "webhook_token": token,
                "webhook_secret": getattr(args, "webhook_secret", None) or "",
            }
    elif robot_type == "app":
        ak = getattr(args, "app_key", None)
        if ak:
            return {
                "name": "__cli__",
                "type": "app",
                "app_key": ak,
                "app_secret": getattr(args, "app_secret", None) or "",
                "robot_code": getattr(args, "robot_code", None) or "",
                "agent_id": getattr(args, "agent_id", None) or "",
            }

    robot = find_robot(config, name=robot_name, robot_type=robot_type)

    if not robot:
        if robot_type == "webhook":
            print("错误：未配置 Webhook")
            print(
                '快速接入：python scripts/dingtalk.py webhook-text --webhook-token <token> "消息内容"'
            )
            print(
                "保存配置：python scripts/dingtalk.py config --set webhook_token=<token>"
            )
        else:
            print("错误：未配置企业内部机器人")
            print(
                "添加机器人：python scripts/dingtalk.py robot-add --name <名称> --type app --app-key <KEY> --app-secret <SECRET>"
            )
        sys.exit(1)

    return robot


def create_messenger_from_robot(robot: dict) -> DingTalkMessenger:
    """从机器人配置创建企业内部机器人发送器"""
    config = DingTalkConfig(
        app_key=robot["app_key"],
        app_secret=robot["app_secret"],
        robot_code=robot.get("robot_code", ""),
        agent_id=robot.get("agent_id", ""),
    )
    return DingTalkMessenger(config)


def create_webhook_from_robot(robot: dict) -> DingTalkWebhookSender:
    """从机器人配置创建 Webhook 发送器"""
    secret = robot.get("webhook_secret", "")
    if secret and secret.startswith("YOUR_"):
        secret = None
    config = DingTalkWebhookConfig(
        webhook_token=robot["webhook_token"],
        secret=secret or None,
    )
    return DingTalkWebhookSender(config)


def send_and_track(robot: dict, send_fn, msg_summary: Optional[str] = None):
    """执行发送并追踪状态，记录消息摘要"""
    name = robot.get("name", "__cli__")
    try:
        result = send_fn()
        success = result.get("errcode", -1) == 0 or "processQueryKey" in result
        if name != "__cli__":
            update_robot_state(name, success=success, msg_summary=msg_summary)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        if name != "__cli__":
            update_robot_state(name, success=False, msg_summary=msg_summary)
        print(f"发送失败: {e}", file=sys.stderr)
        sys.exit(1)


# ==================== 消息命令 ====================


def _make_summary(msg_type: str, content: str = "", title: str = "") -> str:
    """生成消息摘要，用于状态追踪"""
    prefix = f"[{msg_type}]"
    text = title or content or ""
    text = text.replace("\\n", " ").replace("\n", " ").strip()
    if len(text) > 60:
        text = text[:60] + "..."
    return f"{prefix} {text}" if text else prefix


def cmd_send_text(args):
    robot = resolve_robot(args, "app")
    messenger = create_messenger_from_robot(robot)
    user_ids = args.users.split(",") if args.users else None
    summary = _make_summary("text", args.content)

    if args.mode == "o2o":
        if not user_ids:
            print("错误：单聊模式需要提供用户ID (--users)")
            sys.exit(1)
        send_and_track(
            robot,
            lambda: messenger.send_o2o_text(
                user_ids=user_ids,
                content=args.content,
                at_users=args.at_users.split(",") if args.at_users else None,
            ),
            summary,
        )
    else:
        if not args.conversation_id:
            print("错误：群聊模式需要提供会话ID (--conversation-id)")
            sys.exit(1)
        send_and_track(
            robot,
            lambda: messenger.send_group_text(
                open_conversation_id=args.conversation_id,
                content=args.content,
                at_mobiles=args.at_mobiles.split(",") if args.at_mobiles else None,
                at_user_ids=args.at_users.split(",") if args.at_users else None,
                is_at_all=args.at_all,
            ),
            summary,
        )


def cmd_send_markdown(args):
    robot = resolve_robot(args, "app")
    messenger = create_messenger_from_robot(robot)
    user_ids = args.users.split(",") if args.users else None
    summary = _make_summary("markdown", args.content, args.title)

    if args.mode == "o2o":
        if not user_ids:
            print("错误：单聊模式需要提供用户ID (--users)")
            sys.exit(1)
        send_and_track(
            robot,
            lambda: messenger.send_o2o_markdown(
                user_ids=user_ids, title=args.title, text=args.content
            ),
            summary,
        )
    else:
        if not args.conversation_id:
            print("错误：群聊模式需要提供会话ID (--conversation-id)")
            sys.exit(1)
        send_and_track(
            robot,
            lambda: messenger.send_group_markdown(
                open_conversation_id=args.conversation_id,
                title=args.title,
                text=args.content,
                at_mobiles=args.at_mobiles.split(",") if args.at_mobiles else None,
                at_user_ids=args.at_users.split(",") if args.at_users else None,
                is_at_all=args.at_all,
            ),
            summary,
        )


def cmd_send_link(args):
    robot = resolve_robot(args, "app")
    messenger = create_messenger_from_robot(robot)
    user_ids = args.users.split(",") if args.users else None
    summary = _make_summary("link", args.content, args.title)

    if args.mode == "o2o":
        if not user_ids:
            print("错误：单聊模式需要提供用户ID (--users)")
            sys.exit(1)
        send_and_track(
            robot,
            lambda: messenger.send_o2o_link(
                user_ids=user_ids,
                title=args.title,
                text=args.content,
                message_url=args.url,
                pic_url=args.pic_url,
            ),
            summary,
        )
    else:
        if not args.conversation_id:
            print("错误：群聊模式需要提供会话ID (--conversation-id)")
            sys.exit(1)
        send_and_track(
            robot,
            lambda: messenger.send_group_link(
                open_conversation_id=args.conversation_id,
                title=args.title,
                text=args.content,
                message_url=args.url,
                pic_url=args.pic_url,
            ),
            summary,
        )


def cmd_send_action_card(args):
    robot = resolve_robot(args, "app")
    messenger = create_messenger_from_robot(robot)
    user_ids = args.users.split(",") if args.users else None
    summary = _make_summary("action_card", args.content, args.title)

    if args.buttons:
        buttons = []
        for btn in args.buttons.split(";"):
            parts = btn.split(",")
            if len(parts) == 2:
                buttons.append({"title": parts[0], "url": parts[1]})
        if args.mode == "o2o":
            if not user_ids:
                print("错误：单聊模式需要提供用户ID (--users)")
                sys.exit(1)
            send_and_track(
                robot,
                lambda: messenger.send_o2o_action_card_multi(
                    user_ids=user_ids,
                    title=args.title,
                    markdown=args.content,
                    buttons=buttons,
                    btn_orientation=args.btn_orientation,
                ),
                summary,
            )
        else:
            if not args.conversation_id:
                print("错误：群聊模式需要提供会话ID (--conversation-id)")
                sys.exit(1)
            send_and_track(
                robot,
                lambda: messenger.send_group_action_card_multi(
                    open_conversation_id=args.conversation_id,
                    title=args.title,
                    markdown=args.content,
                    buttons=buttons,
                    btn_orientation=args.btn_orientation,
                ),
                summary,
            )
    else:
        if args.mode == "o2o":
            if not user_ids:
                print("错误：单聊模式需要提供用户ID (--users)")
                sys.exit(1)
            send_and_track(
                robot,
                lambda: messenger.send_o2o_action_card(
                    user_ids=user_ids,
                    title=args.title,
                    markdown=args.content,
                    single_title=args.single_title,
                    single_url=args.url,
                ),
                summary,
            )
        else:
            if not args.conversation_id:
                print("错误：群聊模式需要提供会话ID (--conversation-id)")
                sys.exit(1)
            send_and_track(
                robot,
                lambda: messenger.send_group_action_card(
                    open_conversation_id=args.conversation_id,
                    title=args.title,
                    markdown=args.content,
                    single_title=args.single_title,
                    single_url=args.url,
                ),
                summary,
            )


def cmd_send_file(args):
    robot = resolve_robot(args, "app")
    messenger = create_messenger_from_robot(robot)
    user_ids = args.users.split(",") if args.users else None
    if not user_ids:
        print("错误：需要提供用户ID (--users)")
        sys.exit(1)
    summary = _make_summary("file", args.file_name or args.file)
    send_and_track(
        robot,
        lambda: messenger.upload_and_send_o2o_file(
            user_ids=user_ids, file_path=args.file, file_name=args.file_name
        ),
        summary,
    )


def cmd_webhook_text(args):
    robot = resolve_robot(args, "webhook")
    sender = create_webhook_from_robot(robot)
    summary = _make_summary("webhook-text", args.content)
    send_and_track(
        robot,
        lambda: sender.send_text(
            content=args.content,
            at_mobiles=args.at_mobiles.split(",") if args.at_mobiles else None,
            at_user_ids=args.at_users.split(",") if args.at_users else None,
            is_at_all=args.at_all,
        ),
        summary,
    )


def cmd_webhook_markdown(args):
    robot = resolve_robot(args, "webhook")
    sender = create_webhook_from_robot(robot)
    summary = _make_summary("webhook-md", args.content, args.title)
    send_and_track(
        robot,
        lambda: sender.send_markdown(
            title=args.title,
            text=args.content,
            at_mobiles=args.at_mobiles.split(",") if args.at_mobiles else None,
            at_user_ids=args.at_users.split(",") if args.at_users else None,
            is_at_all=args.at_all,
        ),
        summary,
    )


def cmd_webhook_link(args):
    robot = resolve_robot(args, "webhook")
    sender = create_webhook_from_robot(robot)
    summary = _make_summary("webhook-link", args.content, args.title)
    send_and_track(
        robot,
        lambda: sender.send_link(
            title=args.title,
            text=args.content,
            message_url=args.url,
            pic_url=args.pic_url,
        ),
        summary,
    )


def cmd_webhook_action_card(args):
    robot = resolve_robot(args, "webhook")
    sender = create_webhook_from_robot(robot)
    summary = _make_summary("webhook-card", args.content, args.title)
    if args.buttons:
        buttons = []
        for btn in args.buttons.split(";"):
            parts = btn.split(",")
            if len(parts) == 2:
                buttons.append({"title": parts[0], "actionURL": parts[1]})
        send_and_track(
            robot,
            lambda: sender.send_action_card_multi(
                title=args.title,
                text=args.content,
                buttons=buttons,
                btn_orientation=args.btn_orientation,
            ),
            summary,
        )
    else:
        send_and_track(
            robot,
            lambda: sender.send_action_card(
                title=args.title,
                text=args.content,
                single_title=args.single_title,
                single_url=args.url,
                btn_orientation=args.btn_orientation,
            ),
            summary,
        )


def cmd_webhook_feed_card(args):
    robot = resolve_robot(args, "webhook")
    sender = create_webhook_from_robot(robot)
    links = []
    for link_str in args.links.split(";"):
        parts = link_str.split(",")
        if len(parts) >= 2:
            link_item = {"title": parts[0], "messageURL": parts[1]}
            if len(parts) >= 3:
                link_item["picURL"] = parts[2]
            links.append(link_item)
    titles = [l.get("title", "") for l in links[:3]]
    summary = _make_summary("webhook-feed", ", ".join(titles))
    send_and_track(robot, lambda: sender.send_feed_card(links=links), summary)


# ==================== 机器人管理命令 ====================


def cmd_robot_add(args):
    """添加机器人"""
    config = ensure_config()
    robots = config.get("robots", [])

    # 检查名称重复
    for r in robots:
        if r["name"] == args.name:
            print(f"错误：机器人 '{args.name}' 已存在，请使用其他名称或先删除")
            sys.exit(1)

    robot = {"name": args.name, "type": args.type}
    desc = getattr(args, "desc", None) or ""
    if desc:
        robot["description"] = desc

    if args.type == "webhook":
        token = getattr(args, "webhook_token", None)
        if not token:
            print("错误：Webhook 机器人需要提供 --webhook-token")
            sys.exit(1)
        robot["webhook_token"] = token
        robot["webhook_secret"] = getattr(args, "webhook_secret", None) or ""
    elif args.type == "app":
        ak = getattr(args, "app_key", None)
        aks = getattr(args, "app_secret", None)
        if not ak or not aks:
            print("错误：企业内部机器人需要提供 --app-key 和 --app-secret")
            sys.exit(1)
        robot["app_key"] = ak
        robot["app_secret"] = aks
        robot["robot_code"] = getattr(args, "robot_code", None) or ""
        robot["agent_id"] = getattr(args, "agent_id", None) or ""

    robots.append(robot)
    config["robots"] = robots

    # 如果是第一个，设为默认
    if len(robots) == 1 or not config.get("default_robot"):
        config["default_robot"] = args.name

    save_config(config)

    # 初始化状态
    state = load_state()
    if "robots" not in state:
        state["robots"] = {}
    state["robots"][args.name] = {
        "enabled": True,
        "last_used": None,
        "use_count": 0,
        "last_status": None,
        "recent_messages": [],
    }
    save_state(state)

    is_default = config["default_robot"] == args.name
    desc_str = f" - {desc}" if desc else ""
    print(
        f"已添加机器人: {args.name} (类型: {args.type}){desc_str}{' [默认]' if is_default else ''}"
    )
    print(f"  配置文件: {CONFIG_PATH}")


def cmd_robot_list(args):
    """列出所有机器人及状态"""
    config = ensure_config()
    robots = config.get("robots", [])
    state = load_state()
    robot_states = state.get("robots", {})
    default_name = config.get("default_robot")

    if not robots:
        print("暂无机器人配置")
        print(
            "添加机器人：python scripts/dingtalk.py robot-add --name <名称> --type webhook --webhook-token <token>"
        )
        return

    print(f"配置文件: {CONFIG_PATH}")
    print()
    print(f"共 {len(robots)} 个机器人：")
    print()

    for r in robots:
        name = r["name"]
        rtype = r["type"]
        rs = robot_states.get(name, {})
        enabled = rs.get("enabled", True)
        last_used = rs.get("last_used") or "-"
        use_count = rs.get("use_count", 0)
        last_status = rs.get("last_status") or "-"
        is_default = name == default_name

        markers = []
        if is_default:
            markers.append("默认")
        if not enabled:
            markers.append("已禁用")
        marker_str = f" [{', '.join(markers)}]" if markers else ""

        desc = r.get("description", "")
        desc_str = f" - {desc}" if desc else ""
        print(f"  {name}{marker_str}{desc_str}")
        print(f"    类型: {rtype}")

        if rtype == "webhook":
            token = r.get("webhook_token", "")
            if token:
                print(f"    Token: {token[:8]}***")
            has_sign = bool(r.get("webhook_secret"))
            print(f"    加签: {'是' if has_sign else '否'}")
        else:
            print(f"    AppKey: {r.get('app_key', '')[:8]}***")
            print(f"    RobotCode: {r.get('robot_code', '-')}")

        print(f"    使用次数: {use_count}")
        print(f"    最近使用: {last_used}")
        print(f"    最近状态: {last_status}")

        # 显示最近消息摘要
        recent = rs.get("recent_messages", [])
        if recent:
            print(f"    最近消息:")
            for msg in recent[-3:]:
                ts = msg.get("time", "")
                summary = msg.get("summary", "")
                print(f"      [{ts}] {summary}")
        print()


def cmd_robot_remove(args):
    """删除机器人"""
    config = ensure_config()
    robots = config.get("robots", [])

    found = any(r["name"] == args.name for r in robots)
    if not found:
        print(f"错误：找不到机器人 '{args.name}'")
        sys.exit(1)

    config["robots"] = [r for r in robots if r["name"] != args.name]

    if config.get("default_robot") == args.name:
        config["default_robot"] = (
            config["robots"][0]["name"] if config["robots"] else None
        )

    save_config(config)

    # 清理状态
    state = load_state()
    if args.name in state.get("robots", {}):
        del state["robots"][args.name]
        save_state(state)

    print(f"已删除机器人: {args.name}")
    if config.get("default_robot"):
        print(f"当前默认机器人: {config['default_robot']}")


def cmd_robot_default(args):
    """设置默认机器人"""
    config = ensure_config()
    robots = config.get("robots", [])

    found = False
    for r in robots:
        if r["name"] == args.name:
            found = True
            break

    if not found:
        print(f"错误：找不到机器人 '{args.name}'")
        print(f"可用机器人：{', '.join(r['name'] for r in robots)}")
        sys.exit(1)

    config["default_robot"] = args.name
    save_config(config)
    print(f"已将默认机器人设为: {args.name}")


def cmd_robot_enable(args):
    """启用/禁用机器人"""
    config = ensure_config()
    robots = config.get("robots", [])

    found = False
    for r in robots:
        if r["name"] == args.name:
            found = True
            break

    if not found:
        print(f"错误：找不到机器人 '{args.name}'")
        sys.exit(1)

    state = load_state()
    if "robots" not in state:
        state["robots"] = {}
    if args.name not in state["robots"]:
        state["robots"][args.name] = {
            "enabled": True,
            "last_used": None,
            "use_count": 0,
            "last_status": None,
        }

    enabled = not args.disable
    state["robots"][args.name]["enabled"] = enabled
    save_state(state)

    status = "启用" if enabled else "禁用"
    print(f"已{status}机器人: {args.name}")


def cmd_robot_update(args):
    """更新机器人信息（描述等）"""
    config = ensure_config()
    robots = config.get("robots", [])

    target = None
    for r in robots:
        if r["name"] == args.name:
            target = r
            break

    if not target:
        print(f"错误：找不到机器人 '{args.name}'")
        sys.exit(1)

    updated = []
    if args.desc is not None:
        target["description"] = args.desc
        updated.append(f"描述: {args.desc}" if args.desc else "描述: (已清除)")
    if args.rename:
        # 检查新名称是否重复
        for r in robots:
            if r["name"] == args.rename and r is not target:
                print(f"错误：名称 '{args.rename}' 已被使用")
                sys.exit(1)
        old_name = target["name"]
        target["name"] = args.rename
        # 同步更新 default_robot
        if config.get("default_robot") == old_name:
            config["default_robot"] = args.rename
        # 同步更新状态文件中的名称
        state = load_state()
        if old_name in state.get("robots", {}):
            state["robots"][args.rename] = state["robots"].pop(old_name)
            save_state(state)
        updated.append(f"名称: {old_name} -> {args.rename}")

    if not updated:
        print("未指定更新内容，可用参数：--desc, --rename")
        return

    save_config(config)
    print(f"已更新机器人 {args.name}:")
    for u in updated:
        print(f"  {u}")


# ==================== 配置管理命令 ====================


def cmd_config(args):
    """管理配置"""
    sensitive_keys = ("app_secret", "webhook_secret", "webhook_token")

    def mask_value(val):
        """对敏感字段做截断隐藏"""
        if not val or str(val).startswith("YOUR_"):
            return val
        return val[:8] + "***" if len(val) > 8 else "***"

    def mask_config(cfg: dict) -> dict:
        """隐藏配置中的敏感信息"""
        display = json.loads(json.dumps(cfg))
        for key in sensitive_keys:
            if key in display:
                display[key] = mask_value(display[key])
        if "robots" in display:
            for r in display["robots"]:
                for key in sensitive_keys:
                    if key in r:
                        r[key] = mask_value(r[key])
        return display

    if args.show:
        config = load_config()
        if not config:
            print(f"配置文件不存在: {CONFIG_PATH}")
            print()
            print(
                "添加机器人：robot-add --name <名称> --type webhook --webhook-token <token>"
            )
            return

        print(f"配置文件: {CONFIG_PATH}")
        print(json.dumps(mask_config(config), ensure_ascii=False, indent=2))

    elif args.set:
        config = load_config()
        if not config:
            config = {"default_robot": None, "robots": []}
        key, value = args.set.split("=", 1)
        config[key] = value
        save_config(config)
        print(f"已设置 {key}，配置文件: {CONFIG_PATH}")

    elif args.init:
        if CONFIG_PATH.exists() and not args.force:
            print(f"配置文件已存在: {CONFIG_PATH}")
            print("使用 --force 覆盖")
            return
        default_config = {"default_robot": None, "robots": []}
        save_config(default_config)
        print(f"已创建配置文件: {CONFIG_PATH}")
        print(
            "添加机器人：robot-add --name <名称> --type webhook --webhook-token <token>"
        )


# ==================== CLI 参数定义 ====================


def add_robot_arg(parser):
    """添加 --robot 参数"""
    parser.add_argument("--robot", help="指定机器人名称（不指定则使用默认/最近使用的）")


def add_common_args(parser):
    """添加企业内部机器人参数"""
    add_robot_arg(parser)
    parser.add_argument("--app-key", help="应用AppKey（覆盖配置）")
    parser.add_argument("--app-secret", help="应用AppSecret（覆盖配置）")
    parser.add_argument("--robot-code", help="机器人编号（覆盖配置）")
    parser.add_argument("--agent-id", help="应用AgentID（覆盖配置）")


def add_webhook_args(parser):
    """添加 Webhook 参数"""
    add_robot_arg(parser)
    parser.add_argument("--webhook-token", help="Webhook access_token（覆盖配置）")
    parser.add_argument("--webhook-secret", help="加签密钥（覆盖配置）")


def add_webhook_at_args(parser):
    parser.add_argument("--at-users", help="@的用户ID列表，逗号分隔")
    parser.add_argument("--at-mobiles", help="@的手机号列表，逗号分隔")
    parser.add_argument("--at-all", action="store_true", help="@所有人")


def add_message_args(parser):
    parser.add_argument(
        "--mode",
        choices=["o2o", "group"],
        default="o2o",
        help="发送模式：o2o=单聊，group=群聊",
    )
    parser.add_argument("--users", help="用户ID列表，逗号分隔")
    parser.add_argument("--conversation-id", help="群聊会话ID")
    parser.add_argument("--at-users", help="@的用户ID列表，逗号分隔")
    parser.add_argument("--at-mobiles", help="@的手机号列表，逗号分隔")
    parser.add_argument("--at-all", action="store_true", help="@所有人")


def add_robot_credential_args(parser):
    """robot-add 命令的凭证参数"""
    parser.add_argument("--webhook-token", help="Webhook access_token")
    parser.add_argument("--webhook-secret", help="加签密钥")
    parser.add_argument("--app-key", help="应用AppKey")
    parser.add_argument("--app-secret", help="应用AppSecret")
    parser.add_argument("--robot-code", help="机器人编号")
    parser.add_argument("--agent-id", help="应用AgentID")


# ==================== main ====================


def main():
    parser = argparse.ArgumentParser(description="钉钉消息发送工具（支持多机器人管理）")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- 企业内部机器人消息 ----
    text_p = subparsers.add_parser("text", help="发送文本消息（企业内部机器人）")
    add_common_args(text_p)
    add_message_args(text_p)
    text_p.add_argument("content", help="消息内容")

    md_p = subparsers.add_parser("markdown", help="发送Markdown消息（企业内部机器人）")
    add_common_args(md_p)
    add_message_args(md_p)
    md_p.add_argument("--title", required=True, help="消息标题")
    md_p.add_argument("content", help="Markdown内容")

    link_p = subparsers.add_parser("link", help="发送链接消息（企业内部机器人）")
    add_common_args(link_p)
    add_message_args(link_p)
    link_p.add_argument("--title", required=True, help="链接标题")
    link_p.add_argument("content", help="链接描述")
    link_p.add_argument("--url", required=True, help="链接地址")
    link_p.add_argument("--pic-url", help="缩略图URL")

    ac_p = subparsers.add_parser(
        "action-card", help="发送ActionCard消息（企业内部机器人）"
    )
    add_common_args(ac_p)
    add_message_args(ac_p)
    ac_p.add_argument("--title", required=True, help="卡片标题")
    ac_p.add_argument("content", help="卡片内容(Markdown)")
    ac_p.add_argument("--single-title", help="单按钮标题")
    ac_p.add_argument("--url", help="单按钮链接")
    ac_p.add_argument("--buttons", help="多按钮，格式：标题1,链接1;标题2,链接2")
    ac_p.add_argument("--btn-orientation", default="0", help="按钮排列：0=竖排，1=横排")

    file_p = subparsers.add_parser("file", help="发送文件（企业内部机器人）")
    add_common_args(file_p)
    file_p.add_argument("--users", required=True, help="用户ID列表，逗号分隔")
    file_p.add_argument("--file", required=True, help="文件路径")
    file_p.add_argument("--file-name", help="自定义文件名")

    # ---- Webhook 消息 ----
    wt_p = subparsers.add_parser("webhook-text", help="通过 Webhook 发送文本消息")
    add_webhook_args(wt_p)
    add_webhook_at_args(wt_p)
    wt_p.add_argument("content", help="消息内容")

    wm_p = subparsers.add_parser(
        "webhook-markdown", help="通过 Webhook 发送 Markdown 消息"
    )
    add_webhook_args(wm_p)
    add_webhook_at_args(wm_p)
    wm_p.add_argument("--title", required=True, help="消息标题")
    wm_p.add_argument("content", help="Markdown 内容")

    wl_p = subparsers.add_parser("webhook-link", help="通过 Webhook 发送链接消息")
    add_webhook_args(wl_p)
    wl_p.add_argument("--title", required=True, help="链接标题")
    wl_p.add_argument("content", help="链接描述")
    wl_p.add_argument("--url", required=True, help="链接地址")
    wl_p.add_argument("--pic-url", help="缩略图URL")

    wac_p = subparsers.add_parser(
        "webhook-action-card", help="通过 Webhook 发送 ActionCard"
    )
    add_webhook_args(wac_p)
    wac_p.add_argument("--title", required=True, help="卡片标题")
    wac_p.add_argument("content", help="卡片内容(Markdown)")
    wac_p.add_argument("--single-title", help="单按钮标题")
    wac_p.add_argument("--url", help="单按钮链接")
    wac_p.add_argument("--buttons", help="多按钮，格式：标题1,链接1;标题2,链接2")
    wac_p.add_argument("--btn-orientation", default="0", help="按钮排列")

    wfc_p = subparsers.add_parser(
        "webhook-feed-card", help="通过 Webhook 发送 FeedCard"
    )
    add_webhook_args(wfc_p)
    wfc_p.add_argument(
        "--links",
        required=True,
        help="链接列表，格式：标题1,链接1,图片1;标题2,链接2,图片2",
    )

    # ---- 机器人管理 ----
    ra_p = subparsers.add_parser("robot-add", help="添加机器人")
    ra_p.add_argument(
        "--name", required=True, help="机器人名称（唯一标识，如：技术告警群）"
    )
    ra_p.add_argument("--type", required=True, choices=["app", "webhook"], help="类型")
    ra_p.add_argument("--desc", help="机器人描述（用途说明，如：发送告警到技术群）")
    add_robot_credential_args(ra_p)

    rl_p = subparsers.add_parser("robot-list", help="列出所有机器人及状态")

    rr_p = subparsers.add_parser("robot-remove", help="删除机器人")
    rr_p.add_argument("--name", required=True, help="机器人名称")

    ru_p = subparsers.add_parser("robot-update", help="更新机器人信息")
    ru_p.add_argument("--name", required=True, help="机器人名称")
    ru_p.add_argument("--desc", help="更新描述")
    ru_p.add_argument("--rename", help="重命名")

    rd_p = subparsers.add_parser("robot-default", help="设置默认机器人")
    rd_p.add_argument("--name", required=True, help="机器人名称")

    re_p = subparsers.add_parser("robot-enable", help="启用/禁用机器人")
    re_p.add_argument("--name", required=True, help="机器人名称")
    re_p.add_argument("--disable", action="store_true", help="禁用（不加则启用）")

    # ---- 配置管理 ----
    cfg_p = subparsers.add_parser("config", help="管理配置文件")
    cfg_p.add_argument("--show", action="store_true", help="显示当前配置")
    cfg_p.add_argument("--set", help="设置配置项，格式：key=value")
    cfg_p.add_argument("--init", action="store_true", help="初始化配置文件")
    cfg_p.add_argument("--force", action="store_true", help="强制覆盖")

    # ---- 解析 ----
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    CMD_MAP = {
        "text": cmd_send_text,
        "markdown": cmd_send_markdown,
        "link": cmd_send_link,
        "action-card": cmd_send_action_card,
        "file": cmd_send_file,
        "webhook-text": cmd_webhook_text,
        "webhook-markdown": cmd_webhook_markdown,
        "webhook-link": cmd_webhook_link,
        "webhook-action-card": cmd_webhook_action_card,
        "webhook-feed-card": cmd_webhook_feed_card,
        "robot-add": cmd_robot_add,
        "robot-list": cmd_robot_list,
        "robot-remove": cmd_robot_remove,
        "robot-update": cmd_robot_update,
        "robot-default": cmd_robot_default,
        "robot-enable": cmd_robot_enable,
        "config": cmd_config,
    }

    handler = CMD_MAP.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
