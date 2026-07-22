---
name: dingtalk-message
version: 0.3.0
description: 钉钉消息发送技能。支持企业内部机器人（批量单聊/群聊）和 Webhook 自定义机器人两种接入方式，支持多机器人管理，支持文本、Markdown、链接、ActionCard、FeedCard等多种消息类型。
---

# 钉钉消息发送技能

## 概述

支持两种接入方式：
- **Webhook 自定义机器人**：通过 access_token 向群聊发送消息，接入简单
- **企业内部机器人**：通过 app_key/app_secret 发送单聊、群聊消息，功能更全

支持多机器人管理，只有一个时自动使用，多个时按优先级自动选择。

## 环境要求

- Python 3.7+
- `pip install requests`

## 首次配置

### 引导流程

首次使用此技能时，必须按以下流程引导用户完成配置：

1. **询问用户的机器人类型和凭证信息**：
   - Webhook 机器人：需要 `access_token`（和可选的加签密钥 `secret`）
   - 企业内部机器人：需要 `app_key`、`app_secret`、`robot_code`（和可选的 `agent_id`）

2. **执行配置命令**：`python scripts/dingtalk.py robot-add --name "机器人名" --type webhook ...`

3. **验证配置**：`python scripts/dingtalk.py config --show`

### 配置文件路径

配置统一存储在系统配置目录，所有 AI agent 共享，无需重复配置：

| 平台 | 配置文件 | 状态文件 |
|------|---------|---------|
| macOS / Linux | `~/.config/dingtalk/config.json` | `~/.config/dingtalk/state.json` |
| Windows | `%APPDATA%\dingtalk\config.json` | `%APPDATA%\dingtalk\state.json` |

### 手动编辑配置文件

也可直接编辑配置文件，每个机器人用 `name` 标识（建议用群名、用途等有意义的名称），`description` 描述用途，方便智能匹配：

```json
{
    "default_robot": "技术告警群",
    "robots": [
        {
            "name": "技术告警群",
            "type": "webhook",
            "description": "发送技术告警到后端技术群",
            "webhook_token": "你的access_token",
            "webhook_secret": ""
        },
        {
            "name": "内部通知机器人",
            "type": "app",
            "description": "企业内部机器人，支持单聊和群聊",
            "app_key": "你的AppKey",
            "app_secret": "你的AppSecret",
            "robot_code": "你的机器人编号",
            "agent_id": ""
        }
    ]
}
```

> `webhook_token` 填 access_token 即可，脚本自动拼接完整 URL。

### 通过命令行添加

```bash
# 添加 Webhook 机器人
python scripts/dingtalk.py robot-add --name "技术告警群" --type webhook --webhook-token "access_token_xxx" --desc "发送告警到后端技术群"

# 添加企业内部机器人
python scripts/dingtalk.py robot-add --name "内部通知" --type app --app-key dingxxx --app-secret xxx --robot-code robot-xxx --desc "支持单聊群聊"
```

### 验证配置

```bash
python scripts/dingtalk.py config --show
```

## 机器人管理

### 选择逻辑

- 只配置一个机器人时，自动使用
- 多个机器人时：`--robot` 指定 > `default_robot` > 最近使用过的 > 第一个可用的
- 用户未明确指定时，可根据机器人的 `description` 和最近消息记录智能匹配，或询问用户

### 管理命令

```bash
# 添加（--desc 描述用途，便于记忆和智能选择）
python scripts/dingtalk.py robot-add --name "技术告警群" --type webhook --webhook-token "access_token_xxx" --desc "发送告警到后端技术群"
python scripts/dingtalk.py robot-add --name "内部通知" --type app --app-key dingxxx --app-secret xxx --robot-code robot-xxx --desc "企业内部机器人，支持单聊群聊"

# 查看所有机器人（含描述、使用次数、最近消息）
python scripts/dingtalk.py robot-list

# 更新描述 / 重命名
python scripts/dingtalk.py robot-update --name "技术告警群" --desc "后端+SRE告警群"
python scripts/dingtalk.py robot-update --name "alert-bot" --rename "技术告警群"

# 指定机器人发送
python scripts/dingtalk.py webhook-text --robot "技术告警群" "告警消息"

# 设置默认 / 启用禁用 / 删除
python scripts/dingtalk.py robot-default --name "产品日报群"
python scripts/dingtalk.py robot-enable --name "技术告警群" --disable
python scripts/dingtalk.py robot-remove --name "技术告警群"
```

### 使用记录

每次发送消息会自动记录摘要到状态文件，包括：
- 使用次数、最近使用时间、最近状态
- 最近10条消息摘要（消息类型 + 内容前60字）

通过 `robot-list` 可查看，也用于智能选择机器人。

## Webhook 消息

> Webhook 命令以 `webhook-` 前缀开头。
> Webhook URL 获取：钉钉群 > 群设置 > 智能群助手 > 添加自定义机器人 > 复制 Webhook 地址

### 文本消息

```bash
# 已配置时直接发送
python scripts/dingtalk.py webhook-text "消息内容"

# 临时指定 token（无需配置）
python scripts/dingtalk.py webhook-text --webhook-token "access_token_xxx" "消息内容"

# @用户 / @所有人
python scripts/dingtalk.py webhook-text --at-mobiles 13800138000 "消息 @13800138000"
python scripts/dingtalk.py webhook-text --at-all "全员通知"
```

### Markdown 消息

```bash
python scripts/dingtalk.py webhook-markdown \
    --title "天气提醒" \
    "#### 杭州天气\n> 9度，西北风1级"
```

### 链接消息

```bash
python scripts/dingtalk.py webhook-link \
    --title "时代在进步" \
    --url "https://www.dingtalk.com" \
    --pic-url "https://example.com/image.png" \
    "点击查看详情"
```

### ActionCard 消息

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

### FeedCard 消息

```bash
python scripts/dingtalk.py webhook-feed-card \
    --links "新闻1,https://news1.com,https://img1.com/pic.png;新闻2,https://news2.com,https://img2.com/pic.png"
```

### 加签安全

```bash
# 配置加签密钥（一次设置）
python scripts/dingtalk.py config --set webhook_secret=SECxxxxxxxxxxxxxxxxxxxxxxxxxx

# 或通过参数临时指定
python scripts/dingtalk.py webhook-text \
    --webhook-token "access_token_xxx" \
    --webhook-secret "SECxxxxxxxxxxxxxxxxxxxxxxxxxx" \
    "带加签的消息"
```

## 企业内部机器人消息

### 单聊消息

```bash
# 文本
python scripts/dingtalk.py text "Hello!" --users user001,user002

# Markdown
python scripts/dingtalk.py markdown --title "天气提醒" --users user001 \
    "#### 杭州天气\n> 9度，西北风1级"

# 链接
python scripts/dingtalk.py link --title "时代在进步" --url "https://www.dingtalk.com" \
    --users user001 "点击查看详情"

# ActionCard
python scripts/dingtalk.py action-card --title "审批通知" \
    --single-title "查看详情" --url "https://www.dingtalk.com" \
    --users user001 "#### 请假申请\n请审批"

# 文件
python scripts/dingtalk.py file --users user001 --file /path/to/report.pdf --file-name "月度报告.pdf"
```

### 群聊消息

```bash
python scripts/dingtalk.py text "大家好！" \
    --mode group \
    --conversation-id chatxxxxxxxxxxxxxxxx \
    --at-mobiles 13800138000,13900139000
```

## CLI 语法规则

1. `content` 是位置参数（无 `--` 前缀），放在可选参数之后
2. 换行使用 `\n`，脚本自动处理转换
3. `content` 必须用双引号包裹
4. 可通过 `--app-key`、`--robot-code` 等参数临时覆盖配置文件

## 钉钉 Markdown 语法限制

钉钉 Markdown 只支持有限子集，构造内容时必须遵守：

**支持：** 标题(`#`)、加粗(`**`)、链接(`[](url)`)、图片(`![](url)`)、无序列表(`-`)、有序列表(`1.`)、引用(`>`)

**不支持（禁止使用）：** 分隔线(`---`)、表格、代码块、行内代码、删除线、任务列表、斜体、嵌套列表

> 脚本会自动移除分隔线 `---`，但其他不支持的语法需要手动避免。

## 常见错误码

**企业内部机器人：** 40001(token过期) / 40004(无效机器人) / 40009(用户不在可见范围) / 40010(未建立会话) / 40037(发送过频)

**Webhook：** 300001(无效token) / 310000(签名校验失败) / 302503(频率限制，每分钟20条)

## 注意事项

- **频率限制**：企业内部机器人每秒20次，Webhook每分钟20条
- **Webhook 仅群聊**：不支持单聊，link/feedCard 不支持 @功能
- **单聊前置条件**：用户需先主动给机器人发过消息
- **批量限制**：单次最多发送100个用户
- **消息长度**：过长内容建议拆分或使用文件发送
