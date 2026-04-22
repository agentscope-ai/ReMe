"""simple tool call test"""

import json

from reme.core.schema.tool_call import ToolCall


def test_simple_schema():
    """测试简单的工具定义：只有基本类型参数"""
    print("\n========== 测试简单 Schema ==========")

    raw_definition = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "unit": {"type": "string", "description": "温度单位", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    }

    # 解析
    tool_call = ToolCall.model_validate(raw_definition)
    print(f"工具名称: {tool_call.name}")
    print(f"必填参数: {tool_call.parameters.required}")

    # 导出并验证相等性
    dumped_data = tool_call.simple_input_dump()
    print(f"\n原始定义:\n{json.dumps(raw_definition, indent=2, ensure_ascii=False)}")
    print(f"\n导出结果:\n{json.dumps(dumped_data, indent=2, ensure_ascii=False)}")

    # 验证相等
    assert dumped_data == raw_definition, "简单 Schema 导出结果与原始定义不一致"
    print("\n✅ 简单 Schema 测试通过：raw_definition == simple_input_dump()")


def test_medium_nested_schema():
    """测试中等复杂度：包含一层对象嵌套"""
    print("\n========== 测试中等复杂 Schema ==========")

    raw_definition = {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "创建订单",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单ID"},
                    "amount": {"type": "number", "description": "订单金额"},
                    "customer": {
                        "type": "object",
                        "description": "客户信息",
                        "properties": {
                            "name": {"type": "string", "description": "客户姓名"},
                            "email": {"type": "string", "description": "客户邮箱"},
                            "phone": {"type": "string", "description": "联系电话"},
                        },
                        "required": ["name", "email"],
                    },
                },
                "required": ["order_id", "customer"],
            },
        },
    }

    # 解析
    tool_call = ToolCall.model_validate(raw_definition)
    print(f"工具名称: {tool_call.name}")
    print(f"根级必填项: {tool_call.parameters.required}")

    customer_attr = tool_call.parameters.properties["customer"]
    print(f"Customer 子属性: {list(customer_attr.properties.keys())}")
    print(f"Customer 必填项: {customer_attr.required}")

    # 导出并验证相等性
    dumped_data = tool_call.simple_input_dump()
    print(f"\n原始定义:\n{json.dumps(raw_definition, indent=2, ensure_ascii=False)}")
    print(f"\n导出结果:\n{json.dumps(dumped_data, indent=2, ensure_ascii=False)}")

    # 验证相等
    assert dumped_data == raw_definition, "中等复杂 Schema 导出结果与原始定义不一致"
    print("\n✅ 中等复杂 Schema 测试通过：raw_definition == simple_input_dump()")


def test_nested_schema():
    """测试复杂嵌套：包含对象嵌套和数组嵌套"""
    print("\n========== 测试复杂嵌套 Schema ==========")

    # 1. 模拟一个来自 LLM 或 MCP 的复杂嵌套定义
    raw_definition = {
        "type": "function",
        "function": {
            "name": "register_user",
            "description": "注册新用户，包含复杂的元数据和标签",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {"type": "string", "description": "用户名"},
                    "metadata": {
                        "type": "object",
                        "description": "用户元数据",
                        "properties": {
                            "age": {"type": "integer"},
                            "location": {"type": "string"},
                        },
                        "required": ["age"],
                    },
                    "tags": {
                        "type": "array",
                        "description": "用户标签列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tag_id": {"type": "string"},
                                "level": {"type": "number"},
                            },
                            "required": ["tag_id"],
                        },
                    },
                },
                "required": ["username", "metadata"],
            },
        },
    }

    # 2. 解析：将原始字典转化为 ToolCall 实例
    tool_call = ToolCall.model_validate(raw_definition)

    print(f"工具名称: {tool_call.name}")
    print(f"根级必填项: {tool_call.parameters.required}")

    # 验证嵌套深度
    metadata_attr = tool_call.parameters.properties["metadata"]
    print(f"Metadata 子属性: {list(metadata_attr.properties.keys())}")
    print(f"Metadata 必填项: {metadata_attr.required}")

    # 3. 导出：验证 simple_input_dump 是否生成了正确的 JSON Schema
    dumped_data = tool_call.simple_input_dump()

    print(f"\n原始定义:\n{json.dumps(raw_definition, indent=2, ensure_ascii=False)}")
    print(f"\n导出结果:\n{json.dumps(dumped_data, indent=2, ensure_ascii=False)}")

    # 验证相等
    assert dumped_data == raw_definition, "复杂嵌套 Schema 导出结果与原始定义不一致"
    print("\n✅ 复杂嵌套 Schema 测试通过：raw_definition == simple_input_dump()")

    # 4. 转换验证：测试 to_mcp_tool
    mcp_tool = tool_call.to_mcp_tool()
    assert mcp_tool.name == "register_user"
    assert "properties" in mcp_tool.inputSchema["properties"]["tags"]["items"]
    print("✅ 嵌套结构在 MCP Tool 转换中成功保留")


def test_array_of_primitives():
    """测试数组嵌套：数组元素为基本类型"""
    print("\n========== 测试基本类型数组 Schema ==========")

    raw_definition = {
        "type": "function",
        "function": {
            "name": "batch_process",
            "description": "批量处理文件",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "description": "文件路径列表",
                        "items": {"type": "string"},
                    },
                    "priorities": {
                        "type": "array",
                        "description": "优先级列表",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["file_paths"],
            },
        },
    }

    # 解析
    tool_call = ToolCall.model_validate(raw_definition)
    print(f"工具名称: {tool_call.name}")
    print(f"必填参数: {tool_call.parameters.required}")

    file_paths_attr = tool_call.parameters.properties["file_paths"]
    print(f"file_paths 类型: {file_paths_attr.type}")
    t_items_type = file_paths_attr.items.type if hasattr(file_paths_attr.items, "type") else file_paths_attr.items
    print(f"file_paths items 类型: {t_items_type}")

    # 导出并验证相等性
    dumped_data = tool_call.simple_input_dump()
    print(f"\n原始定义:\n{json.dumps(raw_definition, indent=2, ensure_ascii=False)}")
    print(f"\n导出结果:\n{json.dumps(dumped_data, indent=2, ensure_ascii=False)}")

    # 验证相等
    assert dumped_data == raw_definition, "基本类型数组 Schema 导出结果与原始定义不一致"
    print("\n✅ 基本类型数组 Schema 测试通过：raw_definition == simple_input_dump()")


def test_deep_nested_schema():
    """测试深层嵌套：三层以上的嵌套结构"""
    print("\n========== 测试深层嵌套 Schema ==========")

    raw_definition = {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": "创建项目，包含复杂的团队和任务结构",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "项目名称"},
                    "team": {
                        "type": "object",
                        "description": "团队信息",
                        "properties": {
                            "leader": {
                                "type": "object",
                                "description": "团队负责人",
                                "properties": {
                                    "name": {"type": "string"},
                                    "contact": {
                                        "type": "object",
                                        "properties": {
                                            "email": {"type": "string"},
                                            "phone": {"type": "string"},
                                        },
                                        "required": ["email"],
                                    },
                                },
                                "required": ["name", "contact"],
                            },
                            "members": {
                                "type": "array",
                                "description": "团队成员列表",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "role": {"type": "string"},
                                        "skills": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["name", "role"],
                                },
                            },
                        },
                        "required": ["leader"],
                    },
                },
                "required": ["project_name", "team"],
            },
        },
    }

    # 解析
    tool_call = ToolCall.model_validate(raw_definition)
    print(f"工具名称: {tool_call.name}")
    print(f"根级必填项: {tool_call.parameters.required}")

    team_attr = tool_call.parameters.properties["team"]
    leader_attr = team_attr.properties["leader"]
    contact_attr = leader_attr.properties["contact"]
    print(f"Team 必填项: {team_attr.required}")
    print(f"Leader 必填项: {leader_attr.required}")
    print(f"Contact 必填项: {contact_attr.required}")

    # 导出并验证相等性
    dumped_data = tool_call.simple_input_dump()
    print(f"\n原始定义:\n{json.dumps(raw_definition, indent=2, ensure_ascii=False)}")
    print(f"\n导出结果:\n{json.dumps(dumped_data, indent=2, ensure_ascii=False)}")

    # 验证相等
    assert dumped_data == raw_definition, "深层嵌套 Schema 导出结果与原始定义不一致"
    print("\n✅ 深层嵌套 Schema 测试通过：raw_definition == simple_input_dump()")


def test_mixed_types_schema():
    """测试混合类型：包含所有基本类型和嵌套类型"""
    print("\n========== 测试混合类型 Schema ==========")

    raw_definition = {
        "type": "function",
        "function": {
            "name": "configure_system",
            "description": "配置系统参数，包含各种类型",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "description": "是否启用"},
                    "max_connections": {"type": "integer", "description": "最大连接数"},
                    "timeout": {"type": "number", "description": "超时时间（秒）"},
                    "mode": {
                        "type": "string",
                        "description": "运行模式",
                        "enum": ["development", "production", "testing"],
                    },
                    "allowed_ips": {
                        "type": "array",
                        "description": "允许的IP地址列表",
                        "items": {"type": "string"},
                    },
                    "database": {
                        "type": "object",
                        "description": "数据库配置",
                        "properties": {
                            "host": {"type": "string"},
                            "port": {"type": "integer"},
                            "ssl_enabled": {"type": "boolean"},
                        },
                        "required": ["host", "port"],
                    },
                },
                "required": ["enabled", "mode"],
            },
        },
    }

    # 解析
    tool_call = ToolCall.model_validate(raw_definition)
    print(f"工具名称: {tool_call.name}")
    print(f"根级必填项: {tool_call.parameters.required}")

    # 验证各种类型
    print(f"enabled 类型: {tool_call.parameters.properties['enabled'].type}")
    print(f"max_connections 类型: {tool_call.parameters.properties['max_connections'].type}")
    print(f"timeout 类型: {tool_call.parameters.properties['timeout'].type}")
    print(f"mode 枚举值: {tool_call.parameters.properties['mode'].enum}")

    # 导出并验证相等性
    dumped_data = tool_call.simple_input_dump()
    print(f"\n原始定义:\n{json.dumps(raw_definition, indent=2, ensure_ascii=False)}")
    print(f"\n导出结果:\n{json.dumps(dumped_data, indent=2, ensure_ascii=False)}")

    # 验证相等
    assert dumped_data == raw_definition, "混合类型 Schema 导出结果与原始定义不一致"
    print("\n✅ 混合类型 Schema 测试通过：raw_definition == simple_input_dump()")


if __name__ == "__main__":
    test_simple_schema()
    test_medium_nested_schema()
    test_nested_schema()
    test_array_of_primitives()
    test_deep_nested_schema()
    test_mixed_types_schema()
    print("\n" + "=" * 50)
    print("🎉 所有测试用例通过！")
    print("=" * 50)
