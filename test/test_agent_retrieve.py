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
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Query 2 " * 5},
            {"role": "tool", "content": "Large tool output " * 50},
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
    if result:
        print("\n结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def test_compress_mode():
    """测试 compress 模式 - LLM 压缩对话历史"""
    print("\n" + "="*60)
    print("测试 2: Compress 模式")
    print("="*60)

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you!"},
            {"role": "user", "content": "Can you help me with Python?"},
            {"role": "assistant", "content": "Of course! I'd be happy to help with Python."},
            {"role": "user", "content": "What's the latest question?"},
        ],
        "context_manage_mode": "compress",
        "max_total_tokens": 5000,
        "group_token_threshold": 1000,
        "keep_recent_count": 2,
        "store_dir": CACHE_DIR,
        "chat_id": "test_compress"
    }

    result = send_request(payload)
    print(result.json())


def test_auto_mode():
    """测试 auto 模式 - 自动选择压缩策略"""
    print("\n" + "="*60)
    print("测试 3: Auto 模式")
    print("="*60)

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello " * 100},
            {"role": "assistant", "content": "Hi there! " * 50},
            {"role": "tool", "content": "Tool result data " * 200},
            {"role": "user", "content": "Final question " * 50},
        ],
        "context_manage_mode": "auto",
        "max_total_tokens": 100,
        "max_tool_message_tokens": 50,
        "keep_recent_count": 1,
        "store_dir": CACHE_DIR,
        "chat_id": "test_auto"
    }

    result = send_request(payload)
    if result:
        print("\n结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def test_original_example():
    """原始测试用例"""
    print("\n" + "="*60)
    print("测试 6: 原始测试用例")
    print("="*60)

    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. "},
            {"role": "user", "content": "你好，请你帮我检索一下RL Post-training中的训练推理不一致问题"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_6befd06ce58843ba9225c7call_6befd06ce58843ba9225c7",
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
            {"role": "tool", "content": """训练-推理不一致的主要来源
# 3.1 系统设计的根本矛盾
# ● 引擎目标冲突：
#   ○ 推断引擎（如 vLLM、TensorRT-LLM）：追求吞吐量，采用低精度（INT8/FP8）、定制内核（如 speculative decoding）、非确定性计算路径
#   ○ 训练框架（如 FSDP、DeepSpeed）：依赖高精度（FP32/混合精度）、确定性计算，保证梯度稳定
#   ○ 结果：推断与训练阶段的计算路径不一致，导致梯度更新公式存在偏差（∇θvLLM ≠ ∇θFSDP）
# 3.2 数据与内容触发因素
# ● 低概率 Token 生成：
#   ○ 多轮推理、工具调用等场景下，模型被迫生成低概率 Token
#   ○ 推断阶段采样到低概率 Token，训练阶段概率更低，导致重要性权重极低，易触发梯度爆炸（参考 3.3 The Smoking Gun）
# ● 分布外（OOD）内容暴露：
#   ○ 工具响应（如  结构化文本）超出预训练分布，生成更多低概率 Token
#   ○ 多轮交互，非第一轮输出的 mismatch 更高（见图4）
# 3.3 硬件环境变量影响
# ● GPU 架构差异：
#   ○ 不匹配程度：A100 > L20 > H20
#   ○ 物理根源：不同 GPU 的低精度实现、内核优化策略（如 cascade attention）
# ● 动态环境反馈：
#   ○ 相同代码在不同硬件表现差异大（如图8 L20 崩溃，H20 恢复）
# 3.4 训练动态恶化机制
# ● 数值敏感性 → 内核误差放大：
#   ○ RL 优化将参数推向 bfloat16 的极值区
#   ○ 推断与训练引擎的不同计算顺序放大误差，形成螺旋崩溃（见图10）
# ● 自强化崩溃循环：
#   ○ 初始 mismatch 产生高误差 batch，进一步恶化模型状态，持续生成高 mismatch 批次（见图9）"""},
            {"role": "user", "content": "请你对于刚才你说的“训练动态恶化机制”再展开说说！"},
        ],
        "context_manage_mode": "compact",
        "max_total_tokens": 20,
        "max_tool_message_tokens": 10,
        "keep_recent_count": 0,
        "store_dir": CACHE_DIR,
        "chat_id": "research_session_001"
    }

    result = send_request(payload)
    print(result.json())


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

    results = []
    for test_func in tests:
        try:
            result = test_func()
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
    # test_simple_query()
    # test_original_example()
