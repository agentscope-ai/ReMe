"""Test cases for DashScope to AgentScope message conversion."""

import json


def test_plain_text_list_conversion():
    """Test converting a long list of plain text DashScope messages to AgentScope Msgs."""
    from reme.core.utils.agentscope_utils import convert_dashscope_to_agentscope

    print("\n" + "=" * 80)
    print("TEST 1: Plain Text List Conversion (List[Dict] -> List[Msg])")
    print("=" * 80)

    # Long conversation with plain text messages
    dashscope_msgs = [
        {
            "role": "system",
            "content": "你是一个专业的AI助手，擅长回答各种问题。",
        },
        {
            "role": "user",
            "content": "你好！请问你能帮我做什么？",
            "name": "用户A",
        },
        {
            "role": "assistant",
            "content": "你好！我可以帮你回答问题、提供建议、进行对话等。有什么我可以帮助你的吗？",
        },
        {
            "role": "user",
            "content": "我想了解一下今天北京的天气情况。",
            "name": "用户A",
        },
        {
            "role": "assistant",
            "content": "好的，让我帮你查询一下北京的天气。",
            "tool_calls": [
                {
                    "id": "call_weather_001",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "北京", "date": "今天"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_weather_001",
            "name": "get_weather",
            "content": "北京今天天气：晴转多云，气温15-25°C，风力3-4级，空气质量良好，适合户外活动。",
        },
        {
            "role": "assistant",
            "content": "根据天气查询结果，北京今天的天气情况如下：\n- 天气：晴转多云\n- 气温：15-25°C\n- 风力：3-4级\n- 空气质量：良好\n\n今天天气不错，适合户外活动哦！",
        },
        {
            "role": "user",
            "content": "太好了！那你能推荐一些户外活动吗？",
            "name": "用户A",
        },
        {
            "role": "assistant",
            "content": (
                "当然可以！根据今天的天气情况，我推荐以下几个户外活动：\n\n"
                "1. 公园散步或慢跑\n2. 骑自行车游览城市\n3. 去郊外爬山\n"
                "4. 在户外咖啡厅享受阳光\n5. 拍摄城市风景照片\n\n你对哪个活动比较感兴趣呢？"
            ),
        },
        {
            "role": "user",
            "content": "爬山听起来不错！你能推荐几个北京周边的爬山地点吗？",
            "name": "用户A",
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "用户想要北京周边的爬山地点推荐。我应该推荐一些知名且适合休闲爬山的地方，考虑交通便利性和难度适中。",
        },
        {
            "role": "assistant",
            "content": "北京周边有很多适合爬山的好去处，这里给你推荐几个：\n\n**初级难度：**\n1. 香山公园 - 红叶季节尤其美丽\n"
            "2. 景山公园 - 可以俯瞰故宫全景\n\n**中级难度：**\n3. 八达岭长城 - 著名的世界文化遗产\n"
            "4. 慕田峪长城 - 相对人少，风景优美\n\n**进阶难度：**\n5. 妙峰山 - 自然风光秀丽\n"
            "6. 百花山 - 植被丰富，空气清新\n\n建议提前查看开放时间和门票信息，准备好登山装备和充足的水。祝你爬山愉快！",
        },
    ]

    print(f"\n[Input] DashScope messages: {len(dashscope_msgs)} messages")
    print(json.dumps(dashscope_msgs, ensure_ascii=False, indent=2))

    # Convert to AgentScope Msgs
    msgs = convert_dashscope_to_agentscope(dashscope_msgs)

    print(f"\n[Output] AgentScope Msgs: {len(msgs)} messages")
    print("=" * 80)

    for i, msg in enumerate(msgs):
        print(f"\n【Message {i+1}/{len(msgs)}】")
        print(f"  name: {msg.name}")
        print(f"  role: {msg.role}")
        print(f"  content type: {type(msg.content).__name__}")
        print(f"  timestamp: {msg.timestamp}")

        if isinstance(msg.content, str):
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"  content: {content_preview}")
        elif isinstance(msg.content, list):
            print(f"  content blocks: {len(msg.content)} blocks")
            for j, block in enumerate(msg.content):
                block_type = block.get("type")
                print(f"    [{j}] type={block_type}", end="")
                if block_type == "text":
                    text = block.get("text", "")
                    text_preview = text[:60] + "..." if len(text) > 60 else text
                    print(f", text='{text_preview}'")
                elif block_type == "tool_use":
                    print(f", name={block.get('name')}, id={block.get('id')}, input={block.get('input')}")
                elif block_type == "tool_result":
                    output = block.get("output", "")
                    output_preview = output[:60] + "..." if len(output) > 60 else output
                    print(f", name={block.get('name')}, id={block.get('id')}, output='{output_preview}'")
                elif block_type == "thinking":
                    thinking = block.get("thinking", "")
                    thinking_preview = thinking[:60] + "..." if len(thinking) > 60 else thinking
                    print(f", thinking='{thinking_preview}'")
                else:
                    print()

    print("\n" + "=" * 80)
    print("✓ Plain Text List Conversion Test Completed")
    print("=" * 80 + "\n")


def test_multimodal_list_conversion():
    """Test converting a long list of multimodal DashScope messages to AgentScope Msgs."""
    from reme.core.utils.agentscope_utils import convert_dashscope_to_agentscope

    print("\n" + "=" * 80)
    print("TEST 2: Multimodal List Conversion (List[Dict] -> List[Msg])")
    print("=" * 80)

    # Long conversation with multimodal content
    dashscope_msgs = [
        {
            "role": "system",
            "content": "你是一个视觉分析助手，可以分析图片、视频和音频内容。",
        },
        {
            "role": "user",
            "content": [
                {"text": "你好！我想让你帮我分析几张照片。"},
            ],
            "name": "摄影师",
        },
        {
            "role": "assistant",
            "content": "你好！我很乐意帮你分析照片。请上传你想分析的照片。",
        },
        {
            "role": "user",
            "content": [
                {"text": "首先，这是我拍的一张风景照，你觉得构图怎么样？"},
                {
                    "image": "https://img.alicdn.com/imgextra/i1/O1CN01gDEY8M1W114Hi3XcN_"
                    "!!6000000002727-0-tps-1024-406.jpg",
                },
            ],
            "name": "摄影师",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "这张风景照的构图很不错！主要优点包括：\n\n1. 采用了经典的三分法构图\n2. 前景、中景、远景层次分明\n"
                    "3. 色彩饱和度适中，视觉效果舒适\n4. 光线运用得当，明暗对比自然\n\n"
                    "如果要改进的话，可以考虑稍微调整一下地平线的位置。",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"text": "太感谢了！那这两张照片呢？我想对比一下："},
                {"text": "\n第一张："},
                {"image": "https://example.com/photo1_sunrise.jpg"},
                {"text": "\n第二张："},
                {"image": "https://example.com/photo2_sunset.jpg"},
                {"text": "\n它们分别是日出和日落时拍摄的，你觉得哪张效果更好？"},
            ],
            "name": "摄影师",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "让我对比分析一下这两张照片：\n\n**日出照片（第一张）：**\n- 光线柔和，色调偏冷\n"
                    "- 天空呈现淡蓝到橙黄的渐变\n- 画面整体清新明快\n- 适合表现希望和新生的主题\n\n"
                    "**日落照片（第二张）：**\n- 光线温暖，色调偏暖\n- 天空呈现金黄到橙红的渐变\n"
                    "- 画面更有戏剧性和情绪感染力\n"
                    "- 适合表现浪漫和感性的主题\n\n"
                    "两张照片各有特色，难分伯仲。如果是为了表现宁静和希望，推荐日出；如果想营造温馨浪漫的氛围，日落会更好。",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"text": "太专业了！我还拍了一段延时摄影视频，能帮我看看吗？"},
                {
                    "video": [
                        "https://example.com/timelapse/frame001.jpg",
                        "https://example.com/timelapse/frame002.jpg",
                        "https://example.com/timelapse/frame003.jpg",
                        "https://example.com/timelapse/frame004.jpg",
                        "https://example.com/timelapse/frame005.jpg",
                    ],
                },
            ],
            "name": "摄影师",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": (
                        "这段延时摄影很棒！从视频帧可以看出：\n\n"
                        "1. **节奏控制**：时间间隔掌握得很好，云朵移动流畅自然\n"
                        "2. **曝光稳定**：各帧之间曝光一致，没有明显的闪烁\n"
                        "3. **画面稳定**：机位固定得很稳，没有抖动\n"
                        "4. **动态元素**：云朵的流动和光线的变化增加了画面的生动性\n\n"
                        "建议：\n- 可以尝试更长的拍摄时间，捕捉完整的天色变化\n"
                        "- 考虑加入前景元素（如建筑或树木）增加空间感"
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"text": "最后还有一段录音，是我在海边录的环境音，你能听听吗？"},
                {"audio": "https://example.com/audio/beach_ambience.mp3"},
            ],
            "name": "摄影师",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_audio_001",
                    "type": "function",
                    "function": {
                        "name": "analyze_audio",
                        "arguments": (
                            '{"audio_url": "https://example.com/audio/beach_ambience.mp3", '
                            '"analysis_type": "环境音分析"}'
                        ),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_audio_001",
            "name": "analyze_audio",
            "content": (
                "音频分析结果：\n- 音频时长：3分24秒\n- 主要声音：海浪拍岸声、海鸥鸣叫、轻微的风声\n"
                "- 音质：清晰，背景噪音较少\n- 立体声效果：良好，空间感强\n"
                "- 音量平衡：适中，没有过载或削波\n- 频率分布：低频（海浪）和高频（海鸥）分布均衡"
            ),
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": (
                        "你的海边环境音录制得非常专业！根据音频分析：\n\n**技术质量：**\n"
                        "✓ 音质清晰，噪音控制得很好\n✓ 立体声效果出色，能感受到空间感\n✓ 音量平衡适中，听感舒适\n\n"
                        "**内容表现：**\n✓ 成功捕捉了海浪、海鸥、风声等多层次声音\n"
                        "✓ 声音元素丰富但不杂乱\n✓ 很好地营造了宁静放松的海边氛围\n\n**应用建议：**\n"
                        "- 适合用作冥想或放松音乐的背景\n- 可以配合你的海边照片/视频使用\n"
                        "- 建议保留原始文件，方便后期调音\n\n"
                        "总的来说，你在摄影和录音方面都展现了很高的专业水平！"
                    ),
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"text": "非常感谢你详细的分析和建议！这对我帮助很大。"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/thank_you.jpg"},
                },
                {"text": "这是我做的一张感谢卡片，送给你！"},
            ],
            "name": "摄影师",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "谢谢你精美的感谢卡片！很高兴能帮到你。\n\n你的作品都很出色，继续保持这份对摄影和创作的热情！如果以后还有作品想分析或讨论，随时欢迎找我。\n\n祝你创作顺利！📸✨",
                },
            ],
        },
    ]

    print(f"\n[Input] DashScope messages: {len(dashscope_msgs)} messages")
    print(json.dumps(dashscope_msgs, ensure_ascii=False, indent=2))

    # Convert to AgentScope Msgs
    msgs = convert_dashscope_to_agentscope(dashscope_msgs)

    print(f"\n[Output] AgentScope Msgs: {len(msgs)} messages")
    print("=" * 80)

    for i, msg in enumerate(msgs):
        print(f"\n【Message {i+1}/{len(msgs)}】")
        print(f"  name: {msg.name}")
        print(f"  role: {msg.role}")
        print(f"  content type: {type(msg.content).__name__}")
        print(f"  timestamp: {msg.timestamp}")

        if isinstance(msg.content, str):
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"  content: {content_preview}")
        elif isinstance(msg.content, list):
            print(f"  content blocks: {len(msg.content)} blocks")
            for j, block in enumerate(msg.content):
                block_type = block.get("type")
                print(f"    [{j}] type={block_type}", end="")

                if block_type == "text":
                    text = block.get("text", "")
                    text_preview = text[:50] + "..." if len(text) > 50 else text
                    print(f", text='{text_preview}'")
                elif block_type == "image":
                    source = block.get("source", {})
                    url = source.get("url", "")
                    url_preview = url[:50] + "..." if len(url) > 50 else url
                    print(f", url='{url_preview}'")
                elif block_type == "video":
                    source = block.get("source", {})
                    url = source.get("url", "")
                    url_preview = url[:50] + "..." if len(url) > 50 else url
                    print(f", url='{url_preview}'")
                elif block_type == "audio":
                    source = block.get("source", {})
                    url = source.get("url", "")
                    url_preview = url[:50] + "..." if len(url) > 50 else url
                    print(f", url='{url_preview}'")
                elif block_type == "tool_use":
                    print(f", name={block.get('name')}, id={block.get('id')}")
                    print(f"       input={json.dumps(block.get('input'), ensure_ascii=False)}")
                elif block_type == "tool_result":
                    output = block.get("output", "")
                    output_preview = output[:50] + "..." if len(output) > 50 else output
                    print(f", name={block.get('name')}, id={block.get('id')}")
                    print(f"       output='{output_preview}'")
                elif block_type == "thinking":
                    thinking = block.get("thinking", "")
                    thinking_preview = thinking[:50] + "..." if len(thinking) > 50 else thinking
                    print(f", thinking='{thinking_preview}'")
                else:
                    print()

    print("\n" + "=" * 80)
    print("✓ Multimodal List Conversion Test Completed")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Run both tests
    test_plain_text_list_conversion()
    test_multimodal_list_conversion()

    print("\n" + "🎉" * 40)
    print("All tests completed successfully!")
    print("🎉" * 40 + "\n")
