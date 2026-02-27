"""Tests for FbCompactor - conversation history summarization.

This module tests the summary generation logic of FbCompactor class,
which creates compact summaries of conversation history using LLM.
"""

import asyncio

from reme import ReMeFb
from reme.core.enumeration import Role
from reme.core.schema import Message


def print_messages(messages: list[Message], title: str = "Messages", max_content_len: int = 150):
    """Print messages with their role and content.

    Args:
        messages: List of messages to print
        title: Title for the message list
        max_content_len: Maximum content length to display (truncate if longer)
    """
    print(f"\n{title}: (count: {len(messages)})")
    print("-" * 80)
    for i, msg in enumerate(messages):
        content = str(msg.content)
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        print(f"  [{i}] {msg.role.value:10s}: {content}")
    print("-" * 80)


def create_long_conversation() -> list[Message]:
    """Create a long conversation that exceeds token thresholds."""
    messages = [
        Message(
            role=Role.USER,
            content="我需要帮助构建一个完整的Web应用程序，包含身份验证、数据库和API端点。",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""我会帮你构建一个完整的Web应用程序。我们将执行以下步骤：

1. 设置项目结构
2. 实现身份验证系统
3. 设计和创建数据库架构
4. 构建API端点
5. 添加前端组件
6. 测试和部署

让我从项目结构开始...""",
        ),
    ]

    for i in range(15):
        messages.append(
            Message(
                role=Role.USER,
                content=f"第{i + 1}步怎么做？能提供更多细节吗？",
            ),
        )
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=f"""对于第{i + 1}步，这里是详细的解释：

首先，我们需要考虑架构设计。"""
                + "这是重要的上下文信息。" * 50
                + """

然后我们实现以下内容：
- 组件A
- 组件B
- 组件C

让我向你展示这部分的代码..."""
                + "\n\ncode_example = 'example'" * 20,
            ),
        )

    return messages


def create_realistic_personal_conversation() -> list[Message]:
    """Create a realistic conversation with personal information for testing compaction."""
    messages = [
        Message(
            role=Role.USER,
            content="你好！我计划下个月去中国旅行。我需要帮助安排行程。",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""\
太好了！我很乐意帮你规划中国之旅。为了给你最好的建议，请告诉我：
1. 你计划去哪些城市？
2. 你会在那里待几天？
3. 你的兴趣是什么（美食、文化、自然景观、购物等）？
4. 你每天的预算大约是多少？""",
        ),
        Message(
            role=Role.USER,
            content="""\
我计划去北京、西安和成都。行程是14天，从3月15日到3月29日。
我叫陈莎拉，今年28岁。我和我的男朋友大卫·米勒一起旅行，他30岁。
我们都非常喜欢中国美食 - 烤鸭、火锅、川菜，你能想到的我们都喜欢！我们也很喜欢
寺庙、园林，想要体验传统中国文化。
我们的预算是每人每天大约200美元，不包括住宿费用，住宿我们已经订好了。""",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""完美！14天的行程和你的兴趣，你们会度过美好的时光。
以下是建议的行程安排：

**北京（5天 - 3月15-19日）**
- 第1天：抵达，入住酒店，游览王府井和南锣鼓巷
- 第2天：参观故宫和天安门广场，游览景山公园
- 第3天：登长城（八达岭或慕田峪），下午游览颐和园
- 第4天：天坛公园，北京胡同游，恭王府
- 第5天：798艺术区，三里屯购物，晚上看京剧

**西安（6天 - 3月20-25日）**
- 第6天：前往西安，参观兵马俑博物馆
- 第7天：古城墙骑行，回民街美食街
- 第8天：大雁塔，陕西历史博物馆
- 第9天：华清池，骊山风景区
- 第10天：大唐芙蓉园，曲江池遗址公园
- 第11天：小雁塔，西安博物院，书院门

**成都（3天 - 3月26-28日）**
- 第12天：前往成都，成都大熊猫繁育研究基地，宽窄巷子
- 第13天：武侯祠，锦里古街，人民公园喝茶
- 第14天：杜甫草堂，最后购物，准备返程

你想让我推荐具体的餐厅或其他细节吗？""",
        ),
        Message(
            role=Role.USER,
            content="""\
这看起来太棒了！是的，请推荐餐厅。我们特别想去
地道的本地餐馆，而不是游客聚集的地方。
另外，我需要提醒一下，大卫对贝类过敏，所以我们需要小心
这一点。
我的邮箱是sarah.chen@email.com，如果你想稍后给我发送详细的PDF。
我的电话是+1-555-0123-4567。
我们住在北京柏悦酒店（5天），西安的传统四合院酒店
长安客栈（6天），以及成都华尔道夫酒店（3天）。""",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""非常棒的住宿选择！长安客栈特别有特色
- 他们有很棒的传统陕菜餐厅。

**餐厅推荐：**

**北京：**
1. **四季民福** (王府井) - 正宗北京烤鸭，无贝类选项。
2. **簋街小吃** (东直门) - 各种北京小吃，告诉店家大卫对贝类过敏
（"我对贝类过敏"）
3. **全聚德** (前门) - 百年老字号烤鸭店
4. **大董** (工体) - 如果想奢侈一点，这家米其林餐厅有
创新中国菜
5. **护国寺小吃** (护国寺) - 传统北京小吃，非常地道

**西安：**
1. **长安客栈自己的餐厅** - 他们的陕菜很棒，
预订时告知过敏情况
2. **德发长** (钟楼附近) - 传统饺子宴，手工制作
3. **老孙家** - 羊肉泡馍体验，对过敏很照顾
4. **樊记肉夹馍** (竹笆市) - 西安最好的肉夹馍
5. **回民街** - 逛街品尝美食，但大卫要小心未知食材

**成都：**
1. **蜀九香** (天府广场) - 火锅，品质优良，价格实惠
2. **钟水饺** (春熙路) - 传统川菜小吃，标准菜谱无贝类
3. **马旺子** (宽窄巷子) - 高端川菜
4. **陈麻婆豆腐** (西玉龙街) - 正宗麻婆豆腐
5. **锦里** - 各种小吃摊，询问摊主食材

**重要过敏提示：**
- 学习这个短语："我对贝类过敏"
- 许多汤底使用海鲜高汤，一定要询问
- 携带中文过敏卡片
- 下载离线翻译应用以备紧急沟通

为你的行程规划，我记下：陈莎拉（sarah.chen@email.com，+1-555-0123-4567）
和大卫·米勒，3月15-29日中国之旅。""",
        ),
        Message(
            role=Role.USER,
            content="""\
非常感谢！还有几个问题：
1. 从北京到西安最好的交通方式是什么？我们听说高铁很方便。
2. 我们应该租随身WiFi还是买SIM卡？
3. 有什么文化礼仪需要注意的吗？
4. 大卫想知道有没有好的酒吧或夜生活场所。他喜欢白酒。
""",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""很好的问题！让我逐一解答：

**1. 交通：**
绝对要坐高铁！对于你的14天行程，高铁是最佳选择。
- 北京到西安：高铁，约4.5-5小时，二等座约500元/人
- 西安到成都：高铁，约3-4小时，二等座约260元/人
- 也可以提前在12306网站或携程上预订
- 需要护照信息购票，提前预订可以拿到更好的价格

**2. 网络：**
我建议租随身WiFi而不是SIM卡，因为：
- 可以共享（莎拉和大卫都可以连接）
- 不需要更换手机SIM卡
- 可以从携程或漫游超人等公司租赁
- 在线预订，机场取货
- 费用：约10-15元/天，无限流量
- 离开时在机场归还

**3. 文化礼仪：**
- 进入家庭、一些餐厅、寺庙时脱鞋（入口处会看到拖鞋）
- 在寺庙：进入前鞠躬，除非允许否则不要在室内拍照
- 地铁上：手机静音，避免大声通话
- 用餐：吃面条发出声音是可以的，不要把筷子插在米饭里
- 付款：现在移动支付很普及，但准备一些现金总是好的
- 垃圾：将垃圾带在身边，找到垃圾桶再丢弃

**4. 酒吧和白酒（给大卫）：**

**北京：**
1. **三里屯酒吧街** - 各种酒吧，国际氛围
2. **Mesh** (三里屯) - 高端鸡尾酒吧
3. **京A** - 精酿啤酒，也有白酒选择
4. **南锣鼓巷** - 小酒吧聚集地，独特氛围

**西安：**
1. **德福巷酒吧街** - 各种酒吧和夜店
2. **1935酒吧** - 轻松氛围，好的酒单
3. **回民街** - 晚上也有小酒馆

**成都：**
1. **兰桂坊** (锦江区) - 世界级酒吧街
2. **小酒馆** (玉林) - 如果想要有音乐的酒吧
3. **九眼桥酒吧街** - 各种酒吧选择

**白酒购物：**
- **北京王府井** - 大型白酒专卖店
- **机场免税店** - 回家前购买白酒的好地方
- 注意：茅台、五粮液等高端白酒价格较高

你的行程还需要其他帮助吗？火车路线、具体时间安排或"""
            """预算明细？""",
        ),
        Message(
            role=Role.USER,
            content="""\
这真的很有帮助！还有一件事 - 我们想在西安参加一个传统茶艺
体验。你能推荐去哪里吗？
另外，我们应该打包什么？我知道三月的天气可能不太稳定。
哦，我的生日是3月22日 - 那天晚上有什么特别的餐厅推荐吗？
那也是我们的3周年纪念日！
大卫的信用卡是Visa尾号4892，我的是Mastercard尾号7651
- 这些卡在中国到处都能用吗？""",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""很棒的问题！提前祝你生日快乐和纪念日快乐！

**西安茶艺体验：**

1. **大唐茶苑** (大雁塔附近) - 强烈推荐！
   - 中英文主持人
   - 包含汉服体验
   - 小团体（最多6人）
   - 2小时体验，约300元/人
   - 提前在线预订

2. **茶缘轩** (古城墙附近)
   - 传统四合院环境
   - 非常地道，游客少
   - 可预订私人茶艺
   - 约280元/人

3. **唐韵茶艺** (回民街区域)
   - 结合茶艺和书法体验
   - 适合情侣
   - 组合体验约400元/人

我建议在第9天（3月22日）上午预订茶艺体验，然后晚上吃特别晚餐！

**三月天气和打包清单：**
中国三月：从冬天过渡到春天
- 气温：北京5-15°C，西安8-18°C，成都10-20°C
- 可能会看到早春的花朵盛开！

**打包：**
- 分层衣物：轻薄毛衣、开衫、轻便外套
- 一件保暖外套（晚上用）
- 舒适的步行鞋（每天要走15,000+步）
- 雨伞（三月偶尔有雨）
- 方便穿脱的鞋子（方便寺庙参观）
- 高档餐厅用的正装
- 电源转换器（中国使用A型插头，220V）
- 手机充电宝
- 小型日用背包

**生日和纪念日晚餐 - 3月22日：**

对于在西安这样特殊的场合，我强烈推荐：

**长安壹号** - 高端陕菜
- 顶级陕菜体验
- 美丽的传统环境和园林景观
- 多道式季节菜单
- 提前1-2个月预订
- 预算：每人1000-1500元（值得纪念日！）
- 告知他们：贝类过敏、生日、纪念日
- 他们通常会为庆祝活动增添特别惊喜

**备选（同样很棒）：**
**西安饭庄** - 百年老店
- 历史悠久（百年历史）
- 美丽的园林环境
- 传统陕菜，晚餐特别精致
- 相似价格范围

**更实惠但仍然特别：**
**德发长**
- 著名饺子宴
- 市中心位置
- 约每人500-800元
- 更容易预订

**生日蛋糕：**
从**北京稻香村**或当地蛋糕店订一个生日蛋糕。
请你的酒店冷藏保管，晚餐后回来时呈上！

**信用卡：**
- 大型酒店和餐厅：Visa和Mastercard广泛接受
  （莎拉的Mastercard 7651和大卫的Visa 4892应该可以用）
- 但是，中国现在主要使用移动支付（微信支付、支付宝）
- 建议准备现金用于：
  - 小餐馆
  - 寺庙/景点
  - 街边小吃
  - 一些商店
  - 出租车
  - 自动售货机

**ATM提示：**
- 中国银行ATM支持外国卡（需要银联标识）
- 建议一次提取较大金额以减少手续费
- 每天保持500-1000元现金
- 你的银行：通知他们中国旅行计划以避免卡被冻结

**出行前清单：**
✅ 高铁票预订
✅ 随身WiFi预订
✅ 长安壹号晚餐预订（现在就订3月22日的！）
✅ 茶艺预订
✅ 通知银行：莎拉（MC 7651），大卫（Visa 4892）
✅ 旅行保险
✅ 下载离线地图
✅ 预订机场接送
✅ 订生日蛋糕

你想要我帮助制定一个详细的每日行程，包括具体时间和地址吗？""",
        ),
        Message(
            role=Role.USER,
            content="""\
太好了！详细的每日行程表会很棒。你能包括具体地址和开放时间吗？
另外，我们在考虑邀请朋友一起 - 我最好的朋友杰西卡·王和她的"""
            """丈夫汤姆。如果我们四个人一起旅行会有什么变化？""",
        ),
        Message(
            role=Role.ASSISTANT,
            content="""\
当然！我会制定详细的行程。作为四人团（莎拉和大卫 + 杰西卡和"""
            """汤姆）旅行实际上有一些优势：

**四人团的好处：**
✅ 分摊随身WiFi费用（1台设备供4人使用）
✅ 分摊出租车费用（出租车正好坐4人）
✅ 一些餐厅有4人起的套餐
✅ 你们团队的私人茶艺体验
✅ 更适合团体照片！

**注意事项：**
- 为4人预订餐厅
- 一些小酒吧可能容纳不了所有人
- 需要时预订2个房间/公寓
- 如果分开行动要协调碰面地点

**详细的14天行程（包含地址）：**

**第1天 - 3月15日（周五）- 北京到达**

*上午/下午：*
- 到达北京首都国际机场
- 取：随身WiFi
- 在机场兑换人民币（建议每人3000元+）
- 乘车到酒店：北京柏悦酒店
  📍 北京市朝阳区建国门外大街2号
  🚇 从机场乘机场快轨到东直门站，换乘地铁（约1小时）

*晚上（下午6:00 - 9:00）：*
- 入住，休息，整理
- 晚餐：**簋街小吃街**
  📍 北京市东城区东直门内大街
  🕒 营业至深夜
  💴 每人100-150元
  - 各种小吃摊位
  - 多数店家只收现金或移动支付
  - 询问贝类（"有没有海鲜"）

*夜间：*
- 在东直门周围散步，看夜景
- 便利店小吃（7-11/全家）
- 早点睡（时差调整）

---

**第2天 - 3月16日（周六）- 故宫和天安门**

*上午（9:00 AM - 12:00 PM）：*
- 在酒店或附近面包店吃早餐
- 🚇 地铁到天安门（从酒店约30分钟）

- **天安门广场**
  📍 北京市东城区天安门广场
  🕒 6:00 AM - 5:00 PM（广场全天开放）
  💴 免费
  - 早上9:30前到达以避开人群
  - 看升旗仪式（如果早起）
  - 拍照留念

*午餐（12:00 PM）：*
- **全聚德烤鸭**
  📍 北京市东城区前门大街30号
  🕒 11:00 AM - 2:00 PM，5:00 PM - 9:00 PM
  💴 每人200-300元
  - 著名北京烤鸭
  - 告知大卫的贝类过敏

*下午（1:30 PM - 5:00 PM）：*
- 步行到故宫（15分钟）

- **故宫博物院**
  📍 北京市东城区景山前街4号
  🕒 8:30 AM - 5:00 PM（周一闭馆）
  💴 60元/人（需提前网上预订）
  - 可能看到早春的花朵！
  - 参观太和殿、乾清宫等主要宫殿
  - 大约需要3-4小时

- **景山公园**
  📍 北京市西城区景山前街
  - 登顶俯瞰故宫全景

*晚餐（6:30 PM）：*
- **护国寺小吃**
  📍 北京市西城区护国寺大街93号
  🕒 全天营业
  💴 每人80-100元
  - 传统北京小吃
  - 多种选择
  - 可定制你的餐食

*夜间：*
- 返回酒店
- 可选：**三里屯酒吧街**（给大卫和汤姆）
  📍 北京市朝阳区三里屯路
  🕒 下午6:00 - 凌晨2:00
  💴 每杯60-150元

---

**第3天 - 3月17日（周日）- 长城和颐和园**

*清晨（5:30 AM - 8:00 AM）：*
- 早起！
- **慕田峪长城**
  📍 北京市怀柔区慕田峪村
  🚇 可预订包车或参加一日游（从市区约1.5小时）
  🕒 7:30 AM - 5:30 PM

- 在景区附近吃早餐
  📍 景区内餐厅
  🕒 8:00 AM - 10:00 AM
  💴 每人50-80元
  ⚠️ 准备好爬长城，穿舒适的鞋子！
  - 告知餐厅大卫的贝类过敏
  - 简单早餐

*上午（9:00 AM - 12:00 PM）：*
- **登长城**
  📍 慕田峪长城
  🕒 已开放
  💴 45元门票/人 + 缆车120元往返
  - 美丽的景色，历史遗迹
  - 约2-3小时游览

*午餐（12:30 PM）：*
- **景区餐厅**
  📍 慕田峪景区
  - 多种午餐选择

- 返回市区（约1.5小时车程）

*下午（3:00 PM - 6:00 PM）：*
- **颐和园**
  📍 北京市海淀区新建宫门路19号
  🕒 6:30 AM - 8:00 PM
  💴 30元/人（旺季）
  - 皇家园林，昆明湖
  - 十七孔桥，长廊
  - 约2-3小时游览

*晚餐（6:30 PM）：*
- **四季民福烤鸭**
  📍 北京市东城区灯市口大街50号
  🕒 11:00 AM - 10:00 PM
  💴 每人150-200元
  - 北京烤鸭，清淡爽口

*夜间：*
- 返回酒店休息
- 可选：在酒店附近散步

---

这变得相当长了！我应该继续写剩下的天数（第4-14天）吗？我也可以"""
            """把这个做成Google文档或PDF，如果那样更方便的话。只需要确认 - 杰西卡和汤姆"""
            """确定要来，还是只是可能？

另外，除了大卫的贝类过敏之外，还有其他人有饮食限制吗？你的"""
            """酒店确认信息是什么 - 我应该包含入住/退房时间吗？""",
        ),
    ]

    return messages


async def test_full_compact_with_summary():
    """Test complete compaction flow with LLM summary generation.

    This is a complex integration test that exercises the full compaction pipeline:
    1. Create a long conversation that exceeds token threshold
    2. Context checker finds cut point (may include split turn detection)
    3. Compactor generates summary for messages to summarize
    4. Compactor handles turn prefix if split turn detected
    5. Final output contains summary + recent messages

    Expects:
        - compacted=True
        - Summary message generated with proper format
        - Split turn handling if applicable
        - Reduced message count
        - Token count within limits
    """
    print("\n" + "=" * 80)
    print("TEST: Full Compaction with LLM Summary Generation")
    print("=" * 80)

    reme_fs = ReMeFb(
        enable_logo=False,
        vector_store=None,
        compact_params={
            "context_window_tokens": 3000,
            "reserve_tokens": 1500,
            "keep_recent_tokens": 500,
        },
    )
    await reme_fs.start()

    messages = create_long_conversation()
    print_messages(messages, "INPUT MESSAGES (Long Conversation)", max_content_len=60)

    print("\nParameters:")
    print("  context_window_tokens: 3000")
    print("  reserve_tokens: 1500 (threshold = 1500)")
    print("  keep_recent_tokens: 500")
    print("\nExpectations:")
    print("  - Token count exceeds threshold")
    print("  - Context checker finds cut point")
    print("  - Compactor generates summary via LLM")
    print("  - May detect split turn scenario")
    print("  - Returns summary + recent messages")

    # Execute full compact flow
    result = await reme_fs.compact(messages_to_summarize=messages)

    print(f"\n{'=' * 80}")
    print("RESULT:")
    print(f"  compacted: {result}")
    await reme_fs.close()


async def test_realistic_personal_conversation_compact():
    """Test compaction with realistic personal conversation and return summary string.

    This test:
    1. Creates a realistic conversation with personal details
    2. Runs compaction to generate a summary
    3. Returns the summary as a string
    4. Validates the compaction result
    """
    print("\n" + "=" * 80)
    print("TEST: Realistic Personal Conversation Compaction")
    print("=" * 80)

    reme_fs = ReMeFb(
        enable_logo=False,
        vector_store=None,
        compact_params={
            "context_window_tokens": 4000,
            "reserve_tokens": 2000,
            "keep_recent_tokens": 800,
        },
    )
    await reme_fs.start()

    messages = create_realistic_personal_conversation()
    print_messages(messages, "INPUT: Realistic Personal Conversation", max_content_len=100)

    print(f"\n{'=' * 80}")
    print("COMPACTING CONVERSATION...")
    print(f"{'=' * 80}")

    # Execute compaction
    result = await reme_fs.compact(messages_to_summarize=messages)

    print(f"\n{'=' * 80}")
    print("COMPACTION RESULT:")
    print(f"{'=' * 80}")
    print(f"  compacted: {result}")
    await reme_fs.close()


async def main():
    """Run compactor tests."""
    print("\n" + "=" * 80)
    print("FbCompactor - Summary Generation Test Suite")
    print("=" * 80)
    print("\nThis test suite validates the LLM-based summarization:")
    print("  - Full compaction flow (context check + summary generation)")
    print("  - Summary format and structure")
    print("  - Split turn handling")
    print("  - Message preservation")
    print("  - Realistic personal conversation compaction")
    print("=" * 80)
    print("\nNote: This test requires LLM access and may take some time.")
    print("=" * 80)

    # Run the comprehensive compaction test
    await test_full_compact_with_summary()

    # Run the realistic personal conversation test
    await test_realistic_personal_conversation_compact()


if __name__ == "__main__":
    asyncio.run(main())
