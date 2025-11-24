"""
Test file for AgenticRetrieveOp
Simple tests for the React agent with context offload capabilities
"""

import requests
import json
import os

# 配置
BASE_URL = "http://0.0.0.0:8002"
AGENTIC_RETRIEVE_URL = f"{BASE_URL}/agentic_retrieve"
CACHE_DIR = "/Users/zhouwk/PycharmProjects/ReMe/cache_file"


def send_request(payload):
    """发送请求到 agentic_retrieve 端点"""
    headers = {"Content-Type": "application/json"}
    response = requests.post(AGENTIC_RETRIEVE_URL, headers=headers, json=payload)

    return response


def test_compact_mode():
    """测试 compact 模式 - 压缩大工具消息"""
    print("\n" + "="*60)
    print("测试 1: Compact 模式")
    print("="*60)

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Query 1"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "mock_tool",
                        "arguments": "{\"mock_query\": \"search query1\"}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": "response1 " * 50},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "mock_tool",
                        "arguments": "{\"mock_query\": \"search query2\"}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": "response1 " * 50},
            {"role": "user", "content": "Query 3"},
        ],
        "context_manage_mode": "compact",
        "max_total_tokens": 50,
        "max_tool_message_tokens": 10,
        "keep_recent_count": 0,
        "store_dir": CACHE_DIR,
        "chat_id": "test_compact"
    }

    result = send_request(payload)
    return result.json()


def test_compress_mode():
    """测试 compress 模式 - LLM 压缩对话历史"""
    print("\n" + "="*60)
    print("测试 2: Compress 模式")
    print("="*60)

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Query 1"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "mock_tool",
                        "arguments": "{\"mock_query\": \"search query1\"}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": "response1 " * 50},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "mock_tool",
                        "arguments": "{\"mock_query\": \"search query2\"}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": "response1 " * 50},
            {"role": "user", "content": "Query 3"},
        ],
        "context_manage_mode": "compress",
        "max_total_tokens": 10,
        "group_token_threshold": None,
        "keep_recent_count": 2,
        "store_dir": CACHE_DIR,
        "chat_id": "test_compress"
    }

    result = send_request(payload)
    return result.json()


def test_auto_mode():
    """测试 auto 模式 - 自动选择压缩策略"""
    print("\n" + "="*60)
    print("测试 3: Auto 模式")
    print("="*60)

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Query 1"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "mock_tool",
                        "arguments": "{\"mock_query\": \"search query1\"}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": "response1 " * 50},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "mock_tool",
                        "arguments": "{\"mock_query\": \"search query2\"}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": "response1 " * 50},
            {"role": "user", "content": "Query 3"},
        ],
        "context_manage_mode": "auto",
        "max_total_tokens": 100,
        "max_tool_message_tokens": 50,
        "keep_recent_count": 1,
        "store_dir": CACHE_DIR,
        "chat_id": "test_auto"
    }

    result = send_request(payload)
    return result.json()


def test_original_example():
    """原始测试用例"""
    print("\n" + "="*60)
    print("测试 4: 模拟真实场景测试用例")
    print("="*60)

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. 你要习惯于先使用grep，再进行readfile。"},
            {"role": "user", "content": "如何变得更聪明？"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "search_web",
                        "arguments": "{}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": """我们都希望自己能够变得越来越聪明。



而聪明与否的关键，除了取决于我们天生的智商之外，还取决于我们后天，有没有“打磨”自己的大脑。

通常来说，锻炼自己的大脑有两种方式：

1，抽象形式的锻炼；

2，具体形式的锻炼。

例如我们去玩一些益智游戏，就是属于抽象形式的锻炼。

问题是，这种锻炼方式，尽管可以让我们的大脑变得更“敏锐”，但只是让你玩类似益智游戏的时候，更容易上手而已。

这种“聪明”，就无法帮助体现在我们解决现实生活中的问题上了。

所以，我们就需要针对生活特定的目标来锻炼我们的大脑，以获得切实的结果并避免认知能力下降。

记忆练习有四种方法：

"""},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "search_web",
                        "arguments": "{}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": """第一，四细节观察练习；
想要更好地训练自己的大脑，就需要在公众场合进行观察性质的训练，这会提升大脑的灵敏度。

例如，假设某人戴着黑色的帽子，金色的头发，一个三角形的环，一件绿色的毛衣。运用这个方法，就是先观察四个细节，然后再回忆。

一些科学家把这样的大脑训练称为“被动记忆训练”。因为这种锻炼方式是被动的，你并没有使用任何特殊的记忆技巧。

你只是要求你的大脑去做它被设计来做的事情就可以：回忆。

回忆为什么很重要呢？

因为回忆是建立在我们“有效观察”的基础上，如果我们对事物并没有相应的观察，回忆就无法提取信息，从而让大脑的记忆信息和提取信息方面的能力，逐渐下降。

所以，在日常生活中，不管你看到一个事物，碰到一个人，或者经历一件事，运用“四细节观察”这个方法去锻炼自己的大脑，你就能够让自己的大脑在信息整理和
调取方面的能力，变得越来越敏锐。

如果你愿意，你还可以注意到关于建筑物、汽车、电影、书籍和食物的细节，这些细节可以改善记忆力，提高你的认知能力。

当然，在社交活动中，对人的观察才能够提升你的整体能力。毕竟善于观察周围的人，是一项很优秀的社交技能。"""},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "search_web",
                        "arguments": "{}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": """第二，算术训练会提高你的注意力；
一个科学的共识是：算术是一种提高认知能力的强大技能。

这里分享一个简单的，同时可以用来提高逻辑思维能力的方法，就是“加3减7”训练，这是一个有趣的数字心理练习，你可以试试。

开始的时候，你要做的就是随便选一个3位数。然后，把3加到那个数字上3次，然后把结果减去7要7次。

重复该过程至少5次，然后下次选择一个新的3位数字。你也可以采用其他做法，例如以4位数字开头，并使用其他数字进一步挑战你的工作记忆。

例如，你可以从1365开始，加上12，重复12次，然后减去11，重复11次。这取决于你设定的数字来决定你要挑战的等级。

这个算术练习会增强你的工作记忆，因为你需要记住大量的细节来完成它。    """},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "search_web",
                        "arguments": "{}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": """第三，重复和回忆人们在你脑海中说的话
也许你会有这样的感觉：

当我们在说话的时候，并没有人真正在听。那么反过来，我们真的有在认真听别人说的话吗？

你可以训练自己的大脑，让自己更好地专注于别人告诉你的事情，并且尽量记住他们说的每一句话。

这个锻炼方法，初始的做法非常简单:认真倾听别人对你说的话，然后在脑子里重复它们。

例如，想象有人对你说：“明天我要去参加朋友的婚礼，但没想到，我前女友作为伴娘去会去。而我现任的女朋友也想跟着我一起去，说要沾沾喜气这样，
到底不知道会发生什么样的事情。前女友是一个脾气非常暴躁的人，我怕她看到我现在的女朋友，会主动过来说一些阴阳怪气地说话，我真的好怕啊！可能我
又找不到理由不让女朋友去。”

你能够大概回忆这番话不？

要完成这个大脑练习，你所需要做的就是在脑海中默念那个人说的每一句话。这样做可以自动提高你的认知能力，并且记住更多的东西。

信息越有挑战性，你的大脑就会得到越多的锻炼。

值得注意的是，不要让练习变得那么容易，以至于你对它感到厌烦。为了成长，我们都需要大脑训练的挑战。    """},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "search_web",
                        "arguments": "{}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ],
                "tool_call_id": "",
                "time_created": "2025-11-21 14:46:29",
                "metadata": {}
            },
            {"role": "tool", "content": """第四，形象化练习记忆
为了记住更多信息，你可以练习在脑海中创建图像。把要记忆的信息转化为图像，也可以改善我们大脑的认知能力。

例如，你看完一部电影，好比说是《哥斯拉大战金刚》。如果你对一个画面印象特别深刻，你能不能回忆相关的细节呢？

画面上有哪些物品？哥斯拉的动作是什么？金刚又正在做什么？它们正处于什么样的情况？周围还有哪些建筑物等？

这些信息，会在你的大脑中构建出一个具体的画面。画面越是具体，你还原信息的能力就越强，从而有效训练到大脑的记忆里。

形象化记忆，是训练大脑其中一个有效的方法。当你尝试回忆某些事情时，你是从图像中寻找细节，提取信息。

试想一下，当你要记忆“今天要去银行交款”这个信息，如果你把银行想象成是一只张开“血盆大口”的怪兽，然后你战战兢兢地拿着钞票，面对这个怪兽，准备
被它吞噬你的这些钞票，你是不是对此产生强烈的记忆观感呢？

这个形象，就无法让你忘掉这个信息，从而不会因为繁忙的事情而被你忽略了。 """},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7",
                        "type": "function",
                        "name": "search_web",
                        "arguments": "{}",
                        "description": "",
                        "input_schema": {},
                        "output_schema": {}
                    }
                ]
            },
            {"role": "tool", "content": """第五，训练动态恶化机制
训练动态恶化机制，可以提高你的记忆能力。

例如，你正在阅读一本书，并开始阅读某一段内容。你开始阅读时，你正在思考，这个内容是不是很重要，或者这个内容是不是 important enough to re
member.

如果你认为这个内容很重要，那么你开始阅读时，你需要用一种方式来提醒自己，这个内容很重要。"""},
            {"role": "user", "content": "请你就四细节观察练习再展开讲讲，用一个json描述你搜索的结果。"},
        ],
        "context_manage_mode": "compact",
        "max_total_tokens": 200,
        "max_tool_message_tokens": 100,
        "keep_recent_count": 2,
        "store_dir": CACHE_DIR,
        "chat_id": "research_session_001"
    }

    result = send_request(payload)
    return result.json()


def run_all_tests():
    """运行所有测试"""
    # 确保缓存目录存在
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("\n开始运行所有测试...")
    print(f"目标服务: {AGENTIC_RETRIEVE_URL}")

    tests = [
        # test_compact_mode,
        # test_compress_mode,
        # test_auto_mode,
        test_original_example,
    ]
    test_original_example
    results = []
    for test_func in tests:
        try:
            result = test_func()
            print(f"输出结果为：{result}")
            results.append((test_func.__name__, "✓ 成功" if result else "✗ 失败"))
        except Exception as e:
            print(f"\n错误: {e}")
            results.append((test_func.__name__, f"✗ 异常: {str(e)}"))

    # 输出总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_name, status in results:
        print(f"{test_name}: {status}")


if __name__ == "__main__":
    # 可以选择运行单个测试或所有测试

    # 运行所有测试
    run_all_tests()

    # 或者只运行某个特定测试，例如:
    # test_compact_mode()
