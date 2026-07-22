# 钉钉消息发送技能 (dingtalk-message)

钉钉消息发送 CLI 工具，支持**企业内部机器人**和 **Webhook 自定义机器人**两种接入方式，支持文本、Markdown、链接、ActionCard、FeedCard、文件等多种消息类型。

> 官方文档：https://open.dingtalk.com/document/development/development-robot-overview

## 目录

- [API 接口文档](#api-接口文档)
  - [企业内部机器人](#一企业内部机器人-api)
    - [批量发送单聊消息](#1-批量发送人与机器人会话中机器人消息)
    - [发送群聊消息](#2-发送群聊消息)
    - [获取 Access Token](#3-获取-access-token)
    - [上传媒体文件](#4-上传媒体文件)
  - [Webhook 自定义机器人](#二webhook-自定义机器人-api)
    - [Webhook 概述](#webhook-概述)
    - [安全设置](#安全设置)
    - [Webhook 消息类型](#webhook-消息类型)
- [CLI 使用指南](#cli-使用指南)
- [配置说明](#配置说明)
- [消息类型详解](#消息类型详解)
- [Markdown 语法限制](#钉钉-markdown-语法限制)
- [错误码参考](#错误码参考)
- [注意事项与限制](#注意事项与限制)

---

## API 接口文档

### 一、企业内部机器人 API

### 1. 批量发送人与机器人会话中机器人消息

> 官方文档：https://open.dingtalk.com/document/development/chatbots-send-one-on-one-chat-messages-in-batches

#### 接口概述

调用本接口批量发送人与机器人会话（即人与机器人的单聊）中的机器人消息。通过该接口，企业内部应用的机器人可以向多个用户同时发送单聊消息。

#### 请求方式

- **HTTP 方法**：`POST`
- **URL**：`https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend`

#### 请求头

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `x-acs-dingtalk-access-token` | String | 是 | 调用服务端接口的授权凭证（access_token） |
| `Content-Type` | String | 是 | 固定值：`application/json` |

#### 请求体参数（Body）

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `robotCode` | String | 是 | 机器人的编码（robotCode），可在钉钉开发者后台的机器人管理页面获取 |
| `userIds` | List\<String\> | 是 | 接收消息的用户 userId 列表，最多支持 **100** 个 |
| `msgKey` | String | 是 | 消息模板 Key，用于指定消息类型，见下方[消息类型 msgKey 映射表](#消息类型-msgkey-映射表) |
| `msgParam` | String | 是 | 消息模板参数，JSON 字符串格式，内容结构取决于 `msgKey` 的类型 |

#### 消息类型 msgKey 映射表

| 消息类型 | msgKey 值 | 说明 |
|----------|-----------|------|
| 文本消息 | `sampleText` | 纯文本消息，支持 @用户 |
| 图片消息 | `sampleImage` | 需先上传获取 media_id |
| 语音消息 | `sampleVoice` | 需先上传获取 media_id |
| 文件消息 | `sampleFile` | 需先上传获取 media_id |
| 链接消息 | `sampleLink` | 带缩略图的链接卡片 |
| Markdown 消息 | `sampleMarkdown` | 支持有限 Markdown 语法 |
| ActionCard（单按钮） | `sampleActionCard` | 单按钮交互卡片 |
| ActionCard（多按钮） | `sampleMultiActionCard` | 多按钮交互卡片 |


#### 各消息类型 msgParam 结构

**文本消息 (`sampleText`)**

```json
{
  "content": "消息内容",
  "atUserIds": ["userId1", "userId2"]   // 可选，@指定用户
}
```

**Markdown 消息 (`sampleMarkdown`)**

```json
{
  "title": "消息标题",
  "text": "#### 标题\n> 引用内容\n正文"
}
```

**链接消息 (`sampleLink`)**

```json
{
  "title": "链接标题",
  "text": "链接描述",
  "messageUrl": "https://example.com",
  "picUrl": "https://example.com/image.png"   // 可选，缩略图
}
```

**图片消息 (`sampleImage`)**

```json
{
  "mediaId": "@lADPxxxxxxxx",
  "caption": "图片描述"   // 可选
}
```

**文件消息 (`sampleFile`)**

```json
{
  "mediaId": "@lADPxxxxxxxx",
  "fileName": "报告.pdf",
  "fileSize": "1024",      // 可选，单位：字节
  "fileType": "pdf"        // 可选
}
```

**语音消息 (`sampleVoice`)**

```json
{
  "mediaId": "@lADPxxxxxxxx",
  "duration": "10",        // 语音时长，单位：秒
  "fileSize": "2048"       // 可选，单位：字节
}
```

**ActionCard 单按钮 (`sampleActionCard`)**

```json
{
  "title": "卡片标题",
  "markdown": "#### 内容标题\n正文",
  "singleTitle": "查看详情",
  "singleUrl": "https://example.com"
}
```

**ActionCard 多按钮 (`sampleMultiActionCard`)**

```json
{
  "title": "卡片标题",
  "markdown": "#### 内容标题\n正文",
  "btnOrientation": "0",   // "0"=竖排，"1"=横排
  "btns": [
    {"title": "同意", "url": "https://example.com/approve"},
    {"title": "拒绝", "url": "https://example.com/reject"}
  ]
}
```

#### 响应参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `processQueryKey` | String | 消息发送任务的查询 Key，可用于查询发送结果 |

**成功响应示例：**

```json
{
  "processQueryKey": "msgTaskId_xxx"
}
```

**错误响应示例：**

```json
{
  "code": "InvalidParameter.RobotCode",
  "message": "robotCode is invalid",
  "requestid": "xxxx-xxxx-xxxx"
}
```

#### 请求示例（cURL）

```bash
curl -X POST 'https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend' \
  -H 'x-acs-dingtalk-access-token: YOUR_ACCESS_TOKEN' \
  -H 'Content-Type: application/json' \
  -d '{
    "robotCode": "dingxxxxxxxx",
    "userIds": ["user001", "user002"],
    "msgKey": "sampleText",
    "msgParam": "{\"content\": \"Hello, 这是一条测试消息！\"}"
  }'
```

#### 请求示例（Python）

```python
import requests
import json

url = "https://api.dingtalk.com/v1.0/robot/oToMessages/batchSend"
headers = {
    "x-acs-dingtalk-access-token": "YOUR_ACCESS_TOKEN",
    "Content-Type": "application/json"
}
payload = {
    "robotCode": "dingxxxxxxxx",
    "userIds": ["user001", "user002"],
    "msgKey": "sampleText",
    "msgParam": json.dumps({"content": "Hello, 这是一条测试消息！"})
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

---

### 2. 发送群聊消息

> 官方文档：https://open.dingtalk.com/document/orgapp/the-robot-sends-a-group-message

#### 接口概述

调用本接口向指定群聊发送机器人消息。机器人需要先加入群聊才能发送消息。

#### 请求方式

- **HTTP 方法**：`POST`
- **URL**：`https://api.dingtalk.com/v1.0/robot/groupMessages/send`

#### 请求头

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `x-acs-dingtalk-access-token` | String | 是 | access_token |
| `Content-Type` | String | 是 | 固定值：`application/json` |

#### 请求体参数（Body）

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `robotCode` | String | 是 | 机器人编码 |
| `openConversationId` | String | 是 | 群聊会话 ID |
| `msgKey` | String | 是 | 消息模板 Key |
| `msgParam` | String | 是 | 消息模板参数，JSON 字符串 |

#### 响应参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `processQueryKey` | String | 消息发送任务的查询 Key |

---

### 3. 获取 Access Token

> 官方文档：https://open.dingtalk.com/document/orgapp/obtain-the-access_token-of-an-internal-app

#### 请求方式

- **HTTP 方法**：`GET`
- **URL**：`https://oapi.dingtalk.com/gettoken`

#### 查询参数（Query）

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `appkey` | String | 是 | 应用的 AppKey |
| `appsecret` | String | 是 | 应用的 AppSecret |

#### 响应参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `errcode` | Number | 错误码，0 表示成功 |
| `errmsg` | String | 错误信息 |
| `access_token` | String | 访问凭证 |
| `expires_in` | Number | 有效期（秒），通常为 7200 |

#### 响应示例

```json
{
  "errcode": 0,
  "errmsg": "ok",
  "access_token": "xxxxxx",
  "expires_in": 7200
}
```

---

### 4. 上传媒体文件

> 用于发送图片、语音、文件类消息前的文件上传。

#### 请求方式

- **HTTP 方法**：`POST`
- **URL**：`https://oapi.dingtalk.com/media/upload`

#### 查询参数（Query）

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `access_token` | String | 是 | 访问凭证 |

#### 请求体参数（multipart/form-data）

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `type` | String | 是 | 文件类型：`image`/`voice`/`file` |
| `media` | File | 是 | 上传的文件 |

#### 响应参数

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `errcode` | Number | 错误码 |
| `errmsg` | String | 错误信息 |
| `media_id` | String | 媒体文件 ID，后续发送消息时使用 |
| `type` | String | 文件类型 |
| `created_at` | Number | 创建时间戳（毫秒） |

---

### 二、Webhook 自定义机器人 API

> 官方文档：https://open.dingtalk.com/document/orgapp/custom-bot-to-send-group-chat-messages

#### Webhook 概述

自定义机器人是一种可以直接添加到钉钉群聊的机器人，通过 Webhook URL 推送消息到群聊。与企业内部机器人不同，不需要创建应用、不需要 OAuth Token 流程，只需 POST JSON 到 Webhook URL 即可。

**适用场景：**
- 监控告警通知
- CI/CD 构建通知
- 定时数据报告推送
- 简单的群聊消息推送

**Webhook URL 格式：**

```
https://oapi.dingtalk.com/robot/send?access_token=XXXXXX
```

#### 请求方式

- **HTTP 方法**：`POST`
- **Content-Type**：`application/json; charset=utf-8`
- **频率限制**：每个机器人每分钟最多发送 **20** 条消息

#### 安全设置

创建自定义机器人时，必须至少选择以下三种安全方式之一：

**方式一：自定义关键词**

- 最多设置 10 个关键词
- 消息内容必须包含至少一个关键词，否则被拒绝

**方式二：IP 地址白名单**

- 配置允许的 IP 地址或 CIDR 段
- 仅允许白名单内的 IP 发送消息

**方式三：加签（推荐）**

启用加签后，每次请求需要在 URL 中附加 `timestamp` 和 `sign` 参数。

**签名算法：**

```
timestamp = 当前毫秒时间戳
string_to_sign = timestamp + "\n" + secret
sign = URL_Encode(Base64(HMAC-SHA256(secret, string_to_sign)))
```

**签名后的 URL：**

```
https://oapi.dingtalk.com/robot/send?access_token=XXXXXX&timestamp=1609459200000&sign=YYYY
```

> 时间戳与服务器时间差不能超过 1 小时，否则请求被拒绝。
> 脚本已内置加签逻辑，只需在配置文件中设置 `webhook_secret` 即可自动签名。

**Python 签名示例：**

```python
import time, hmac, hashlib, base64, urllib.parse

def generate_sign(secret: str):
    timestamp = str(round(time.time() * 1000))
    string_to_sign = f'{timestamp}\n{secret}'
    hmac_code = hmac.new(
        secret.encode('utf-8'),
        string_to_sign.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
    return timestamp, sign
```

#### Webhook 消息类型

##### 文本消息 (text)

```json
{
    "msgtype": "text",
    "text": {
        "content": "消息内容"
    },
    "at": {
        "atMobiles": ["13800138000"],
        "atUserIds": ["user123"],
        "isAtAll": false
    }
}
```

##### Markdown 消息 (markdown)

```json
{
    "msgtype": "markdown",
    "markdown": {
        "title": "标题（通知栏显示）",
        "text": "#### 标题\n> 引用内容\n正文"
    },
    "at": {
        "atMobiles": ["13800138000"],
        "isAtAll": false
    }
}
```

##### 链接消息 (link)

> 不支持 @功能

```json
{
    "msgtype": "link",
    "link": {
        "title": "链接标题",
        "text": "链接描述",
        "messageUrl": "https://example.com",
        "picUrl": "https://example.com/image.png"
    }
}
```

##### ActionCard 单按钮 (actionCard)

```json
{
    "msgtype": "actionCard",
    "actionCard": {
        "title": "卡片标题",
        "text": "#### 内容\n正文（支持 Markdown）",
        "btnOrientation": "0",
        "singleTitle": "阅读全文",
        "singleURL": "https://example.com/"
    }
}
```

##### ActionCard 多按钮 (actionCard)

```json
{
    "msgtype": "actionCard",
    "actionCard": {
        "title": "卡片标题",
        "text": "#### 内容\n正文",
        "btnOrientation": "0",
        "btns": [
            {"title": "同意", "actionURL": "https://example.com/approve"},
            {"title": "拒绝", "actionURL": "https://example.com/reject"}
        ]
    }
}
```

> 注意：多按钮时使用 `btns` + `actionURL`，不要同时设置 `singleTitle`/`singleURL`

##### FeedCard 消息 (feedCard)

> 不支持 @功能

```json
{
    "msgtype": "feedCard",
    "feedCard": {
        "links": [
            {
                "title": "链接标题1",
                "messageURL": "https://example.com/1",
                "picURL": "https://example.com/pic1.png"
            },
            {
                "title": "链接标题2",
                "messageURL": "https://example.com/2",
                "picURL": "https://example.com/pic2.png"
            }
        ]
    }
}
```

#### Webhook 响应

**成功：**

```json
{
    "errcode": 0,
    "errmsg": "ok"
}
```

**错误示例：**

```json
{
    "errcode": 310000,
    "errmsg": "keywords not in content"
}
```

#### 请求示例（cURL）

```bash
curl -X POST 'https://oapi.dingtalk.com/robot/send?access_token=XXXXXX' \
  -H 'Content-Type: application/json; charset=utf-8' \
  -d '{
    "msgtype": "text",
    "text": {
        "content": "Hello, 这是一条 Webhook 测试消息！"
    }
  }'
```

#### Webhook 与企业内部机器人对比

| 维度 | Webhook 自定义机器人 | 企业内部机器人 |
|------|---------------------|---------------|
| **接入方式** | 群聊添加自定义机器人 | 钉钉开放平台创建应用 |
| **API 域名** | `oapi.dingtalk.com/robot/send` | `api.dingtalk.com/v1.0/robot/` |
| **认证方式** | URL 中的 access_token + 可选签名 | Header 中的 OAuth access_token |
| **配置项** | webhook_url, webhook_secret | app_key, app_secret, robot_code |
| **消息范围** | 仅所在群聊 | 任意用户/群聊 |
| **消息格式** | `msgtype` + 类型对象 | `msgKey` + `msgParam` (JSON 字符串) |
| **频率限制** | 20 条/分钟 | 20 次/秒 |
| **Token 管理** | 无需刷新（URL 固定） | Token 每 2 小时过期 |

---

## CLI 使用指南

### 安装依赖

```bash
pip install requests
```

### 命令行格式

```bash
# 企业内部机器人
python scripts/dingtalk.py <消息类型> [选项参数] "<消息内容>" --users <用户ID>

# Webhook 自定义机器人（命令以 webhook- 前缀开头）
python scripts/dingtalk.py webhook-<消息类型> [选项参数] "<消息内容>"
```

### 企业内部机器人命令

#### 发送文本消息

```bash
# 单聊
python scripts/dingtalk.py text "Hello, 这是一条测试消息！" --users user001,user002

# @指定用户
python scripts/dingtalk.py text "Hello, @user001" --users user001,user002 --at-users user001

# 群聊
python scripts/dingtalk.py text "大家好！" \
    --mode group \
    --conversation-id chatxxxxxxxxxxxxxxxx \
    --at-mobiles 13800138000,13900139000
```

#### 发送 Markdown 消息

```bash
python scripts/dingtalk.py markdown \
    --title "天气提醒" \
    --users user001 \
    "#### 杭州天气\n> 9度，西北风1级"
```

#### 发送链接消息

```bash
python scripts/dingtalk.py link \
    --title "时代在进步" \
    --url "https://www.dingtalk.com" \
    --pic-url "https://example.com/image.png" \
    --users user001 \
    "点击查看详情"
```

#### 发送 ActionCard 消息

```bash
# 单按钮
python scripts/dingtalk.py action-card \
    --title "审批通知" \
    --single-title "查看详情" \
    --url "https://www.dingtalk.com" \
    --users user001 \
    "#### 请假申请\n请审批"

# 多按钮
python scripts/dingtalk.py action-card \
    --title "审批通知" \
    --buttons "同意,https://approve.com/yes;拒绝,https://approve.com/no" \
    --btn-orientation 0 \
    --users user001 \
    "#### 请假申请"
```

#### 发送文件

```bash
python scripts/dingtalk.py file \
    --users user001,user002 \
    --file /path/to/report.pdf \
    --file-name "月度报告.pdf"
```

### Webhook 自定义机器人命令

> 所有 Webhook 命令以 `webhook-` 前缀开头。
> Webhook URL 可通过 `--webhook-url` 参数传入，或在配置文件中设置 `webhook_url`。

#### Webhook 发送文本消息

```bash
# 使用配置文件中的 webhook_url
python scripts/dingtalk.py webhook-text "Hello, 这是一条 Webhook 消息！"

# 通过参数指定 webhook URL
python scripts/dingtalk.py webhook-text \
    --webhook-url "https://oapi.dingtalk.com/robot/send?access_token=xxx" \
    "Hello, 测试消息"

# @指定手机号
python scripts/dingtalk.py webhook-text \
    --at-mobiles 13800138000,13900139000 \
    "通知内容 @13800138000"

# @所有人
python scripts/dingtalk.py webhook-text --at-all "全员通知"
```

#### Webhook 发送 Markdown 消息

```bash
python scripts/dingtalk.py webhook-markdown \
    --title "天气提醒" \
    "#### 杭州天气\n> 9度，西北风1级"
```

#### Webhook 发送链接消息

```bash
python scripts/dingtalk.py webhook-link \
    --title "时代在进步" \
    --url "https://www.dingtalk.com" \
    --pic-url "https://example.com/image.png" \
    "点击查看详情"
```

#### Webhook 发送 ActionCard 消息

```bash
# 单按钮
python scripts/dingtalk.py webhook-action-card \
    --title "审批通知" \
    --single-title "查看详情" \
    --url "https://www.dingtalk.com" \
    "#### 请假申请\n请审批"

# 多按钮
python scripts/dingtalk.py webhook-action-card \
    --title "审批通知" \
    --buttons "同意,https://approve.com/yes;拒绝,https://approve.com/no" \
    --btn-orientation 0 \
    "#### 请假申请"
```

#### Webhook 发送 FeedCard 消息

```bash
python scripts/dingtalk.py webhook-feed-card \
    --links "新闻标题1,https://news1.com,https://img1.com/pic.png;新闻标题2,https://news2.com,https://img2.com/pic.png"
```

#### Webhook 使用加签

```bash
# 通过命令行参数
python scripts/dingtalk.py webhook-text \
    --webhook-url "https://oapi.dingtalk.com/robot/send?access_token=xxx" \
    --webhook-secret "SECxxxxxxxxxxxxxxxxxxxxxxxxxx" \
    "带签名的消息"

# 或在配置文件中设置（推荐）
python scripts/dingtalk.py config --set webhook_secret=SECxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 查看帮助

```bash
python scripts/dingtalk.py --help
python scripts/dingtalk.py text --help
python scripts/dingtalk.py webhook-text --help
python scripts/dingtalk.py webhook-markdown --help
```

---

## 配置说明

### 配置文件

配置统一存储在系统配置目录，所有 AI agent 共享，无需重复配置：

| 平台 | 配置文件 | 状态文件 |
|------|---------|---------|
| macOS / Linux | `~/.config/dingtalk/config.json` | `~/.config/dingtalk/state.json` |
| Windows | `%APPDATA%\dingtalk\config.json` | `%APPDATA%\dingtalk\state.json` |

```json
{
    "default_robot": "技术告警群",
    "robots": [
        {
            "name": "技术告警群",
            "type": "webhook",
            "description": "发送技术告警到后端技术群",
            "webhook_token": "YOUR_ACCESS_TOKEN",
            "webhook_secret": ""
        },
        {
            "name": "内部通知机器人",
            "type": "app",
            "description": "企业内部机器人，支持单聊和群聊",
            "app_key": "YOUR_APP_KEY",
            "app_secret": "YOUR_APP_SECRET",
            "robot_code": "YOUR_ROBOT_CODE",
            "agent_id": "YOUR_AGENT_ID"
        }
    ]
}
```

### 参数获取说明

#### 企业内部机器人

| 参数 | 说明 | 获取方式 | 必填 |
|------|------|----------|------|
| `app_key` | 应用 AppKey | 钉钉开放平台 > 应用详情 > 凭证与基础信息 | 是 |
| `app_secret` | 应用 AppSecret | 钉钉开放平台 > 应用详情 > 凭证与基础信息 | 是 |
| `robot_code` | 机器人编号 | 钉钉开放平台 > 应用详情 > 机器人与消息推送 | 是 |
| `agent_id` | 应用 AgentID | 钉钉开放平台 > 应用详情 > 凭证与基础信息 | 否 |

#### Webhook 自定义机器人

| 参数 | 说明 | 获取方式 | 必填 |
|------|------|----------|------|
| `webhook_url` | Webhook 地址 | 钉钉群 > 群设置 > 智能群助手 > 添加自定义机器人 > 复制 Webhook | 是 |
| `webhook_secret` | 加签密钥 | 创建机器人时选择"加签"安全方式，复制 SEC 开头的密钥 | 否* |

> *如果安全设置选择了"加签"方式，则 webhook_secret 必填。
> 两种机器人的配置可以同时存在于同一个配置文件中，互不影响。

### 配置管理命令

```bash
# 初始化配置文件
python scripts/dingtalk.py config --init

# 查看当前配置
python scripts/dingtalk.py config --show

# 设置单个配置项
python scripts/dingtalk.py config --set app_key=dingxxxxxxxxxxxx

# 添加机器人
python scripts/dingtalk.py robot-add --name "技术告警群" --type webhook --webhook-token "token_xxx"
python scripts/dingtalk.py robot-add --name "内部通知" --type app --app-key dingxxx --app-secret xxx --robot-code robot-xxx

# 强制覆盖已存在的配置
python scripts/dingtalk.py config --init --force
```

### 命令行参数覆盖

配置文件中的值可以通过命令行参数临时覆盖：

```bash
python scripts/dingtalk.py text "测试消息" \
    --users user001 \
    --app-key dingxxxxxxxxxxxx \
    --app-secret your-secret-key \
    --robot-code robot-xxxxxx
```

---

## 消息类型详解

| 类型 | CLI 指令 | 模式 | 说明 |
|------|----------|------|------|
| 文本消息 | `text` | 单聊/群聊 | 纯文本，支持 @用户 |
| Markdown | `markdown` | 单聊/群聊 | 有限 Markdown 语法的富文本 |
| 链接消息 | `link` | 单聊/群聊 | 带缩略图的链接卡片 |
| ActionCard | `action-card` | 单聊/群聊 | 带按钮的交互卡片（单/多按钮） |
| 文件消息 | `file` | 仅单聊 | 发送文件，自动上传 |
| 图片消息 | `image` | 仅单聊 | 发送图片 |
| 语音消息 | `voice` | 仅单聊 | 发送语音 |


---

## 钉钉 Markdown 语法限制

钉钉的 Markdown 渲染器只支持标准 Markdown 的有限子集。

### 支持的语法

| 语法 | 写法 | 说明 |
|------|------|------|
| 标题 | `# 一级标题` ~ `###### 六级标题` | 正常支持 |
| 加粗 | `**粗体文字**` | 正常支持 |
| 链接 | `[链接文字](url)` | 正常支持 |
| 图片 | `![alt](图片url)` | 正常支持 |
| 无序列表 | `- 列表项` | 正常支持 |
| 有序列表 | `1. 列表项` | 正常支持 |
| 引用 | `> 引用文字` | 正常支持 |

### 不支持的语法（禁止使用）

以下语法在钉钉中**不会被渲染**，甚至可能导致显示异常：

- 水平分隔线：`---`、`***`、`___`
- 表格：`| col1 | col2 |`
- 代码块：`` ``` ``
- 行内代码：`` `code` ``
- 删除线：`~~text~~`
- 任务列表：`- [ ] item`
- 斜体：`*italic*`（不稳定）
- 嵌套列表：多级缩进列表（显示不稳定）

### 编写建议

1. 用 `#` ~ `####` 标题组织结构
2. 用 `- item` 无序列表展示条目
3. 用 `**重点**` 加粗强调关键信息
4. 用 `> 备注` 添加补充说明
5. 不要用分隔线，用标题或空行替代
6. 不要用表格，用列表格式替代
7. 在 CLI 中使用 `\n` 表示换行

---

## 错误码参考

### 企业内部机器人

| 错误码 | 说明 |
|--------|------|
| 0 | 成功 |
| -1 | 系统繁忙，请稍后重试 |
| 40001 | access_token 不存在或已过期 |
| 40002 | access_token 不合法 |
| 40004 | 无效的机器人（robotCode 错误） |
| 40007 | 无效的 openConversationId |
| 40008 | 无效的消息内容（msgParam 格式错误） |
| 40009 | 消息发送失败，用户不在该机器人可见范围 |
| 40010 | 消息发送失败，用户未与机器人建立会话 |
| 40014 | 无效的 userId |
| 40037 | 发送消息过于频繁，已触发限流 |
| 40056 | 无效的 agentId |

### Webhook 自定义机器人

| 错误码 | 说明 |
|--------|------|
| 0 | 成功 |
| 300001 | 无效的 token / 机器人不存在 |
| 310000 | 安全校验失败（关键词不匹配 / IP 不在白名单 / 签名错误 / 时间戳过期） |
| 302503 | 频率限制，每分钟超过 20 条 |
| 400013 | JSON 格式无效 |

---

## 注意事项与限制

### 企业内部机器人

#### 频率限制

- 每个应用每秒钟最多调用 **20 次** 消息发送接口
- 超出限制会返回错误码 `40037`

#### 用户限制

- 批量发送单聊消息时，`userIds` 每次最多 **100 个**
- 超过 100 个需分批发送

#### 会话前置条件

- 发送单聊消息前，用户需要先与机器人**建立会话**（用户主动给机器人发送过消息）
- 未建立会话的用户会收到错误码 `40010`

#### 文件上传

- 发送文件/图片/语音消息前，需要先通过 `/media/upload` 接口上传文件获取 `media_id`
- CLI 工具的 `file` 命令已自动集成上传流程

#### Token 管理

- `access_token` 有效期为 **7200 秒**（2 小时）
- 工具内置缓存机制，过期前 5 分钟自动刷新
- 不建议频繁调用 gettoken 接口

### Webhook 自定义机器人

#### 频率限制

- 每个机器人每分钟最多发送 **20 条** 消息
- 超出限制返回错误码 `302503`

#### 安全设置

- 创建机器人时必须至少选择一种安全方式：自定义关键词、IP 白名单、加签
- 使用加签时，时间戳与服务器时间差不能超过 **1 小时**
- 脚本内置加签逻辑，配置 `webhook_secret` 后自动签名

#### 消息范围

- Webhook 机器人只能在添加到的群聊中发送消息，**不支持单聊**
- `link` 和 `feedCard` 消息类型**不支持** @功能

#### Token 管理

- Webhook URL 中的 `access_token` 是固定的，无需刷新
- 只要机器人未被删除，URL 持续有效

### 通用

#### 消息内容

- Markdown 消息仅支持有限语法子集，见 [Markdown 语法限制](#钉钉-markdown-语法限制)
- 消息内容不宜过长，过长建议拆分或使用文件发送
- 企业内部机器人的 `msgParam` 需要序列化为 JSON 字符串后传入（脚本已自动处理）

#### API 域名

- 企业内部机器人（消息发送）：`https://api.dingtalk.com`
- 企业内部机器人（gettoken、文件上传）：`https://oapi.dingtalk.com`
- Webhook 自定义机器人：`https://oapi.dingtalk.com/robot/send`
