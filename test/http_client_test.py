import asyncio
import json

import aiohttp

base_url = "http://0.0.0.0:8002"


async def run1(session):
    workspace_id = "default1"

    async with session.post(
        f"{base_url}/vector_store",
        json={
            "action": "delete",
            "workspace_id": workspace_id,
        },
        headers={"Content-Type": "application/json"},
    ) as response:
        result = await response.json()
        print(json.dumps(result, ensure_ascii=False))

    trajectories = [
        {
            "task_id": "t1",
            "messages": [
                {"role": "user", "content": "搜索可以使用websearch工具"},
            ],
            "score": 1,
        },
        {
            "task_id": "t1",
            "messages": [
                {"role": "user", "content": "搜索可以使用code工具"},
            ],
            "score": 0,
        },
    ]

    async with session.post(
        # f"{base_url}/summary_task_memory",
        f"{base_url}/summary_task_memory_simple",
        json={
            "trajectories": trajectories,
            "workspace_id": workspace_id,
        },
        headers={"Content-Type": "application/json"},
    ) as response:
        result = await response.json()
        print(json.dumps(result, ensure_ascii=False))

    await asyncio.sleep(2)

    async with session.post(
        # f"{base_url}/retrieve_task_memory",
        f"{base_url}/retrieve_task_memory_simple",
        json={
            "query": "茅台怎么样？",
            "workspace_id": workspace_id,
        },
        headers={"Content-Type": "application/json"},
    ) as response:
        result = await response.json()
        print(json.dumps(result, ensure_ascii=False))


async def run2(session):
    workspace_id = "default2"

    async with session.post(
        f"{base_url}/vector_store",
        json={
            "action": "delete",
            "workspace_id": workspace_id,
        },
        headers={"Content-Type": "application/json"},
    ) as response:
        result = await response.json()
        print(json.dumps(result, ensure_ascii=False))

    messages = [
        {"role": "user", "content": "我喜欢吃西瓜🍉"},
        {"role": "user", "content": "昨天吃了苹果，很好吃"},
        {"role": "user", "content": "我不太喜欢吃西瓜"},
        {"role": "user", "content": "上周我去了日本，得了肠胃炎"},
        {"role": "user", "content": "这周只能在家里，喝粥"},
    ]

    async with session.post(
        f"{base_url}/summary_personal_memory",
        json={
            "messages": messages,
            "workspace_id": workspace_id,
        },
        headers={"Content-Type": "application/json"},
    ) as response:
        result = await response.json()
        print(json.dumps(result, ensure_ascii=False))

    await asyncio.sleep(2)

    async with session.post(
        f"{base_url}/retrieve_personal_memory",
        json={
            "query": "你知道我喜欢吃什么？",
            "workspace_id": workspace_id,
        },
        headers={"Content-Type": "application/json"},
    ) as response:
        result = await response.json()
        print(json.dumps(result, ensure_ascii=False))


async def run3(session):
    workspace_id = "default2"

    async with session.post(
        f"{base_url}/add_tool_call_result",
        json={
            "tool_call_results": [
                {"a": 1},
                {"a": 2},
            ],
            "workspace_id": workspace_id,
        },
        headers={"Content-Type": "application/json"},
    ) as response:
        result = await response.json()
        print(json.dumps(result, ensure_ascii=False))


async def run4(session):
    workspace_id = "default4"

    async with session.post(
            f"{base_url}/agentic_retrieve",
            json={
                "messages": [
                    {"role": "user", "content": "hello" * 10000},
                ],
                "workspace_id": workspace_id,
                "context_manage_mode": "auto",
                "keep_recent_count": 0,
                "max_total_tokens": 10000,
            },
            headers={"Content-Type": "application/json"},
    ) as response:
        result = await response.json()
        print(json.dumps(result, ensure_ascii=False))


async def main():

    async with aiohttp.ClientSession() as session:
        # 获取工具列表
        print("获取工具列表...")

        # await run1(session)
        # await run2(session)
        # await run3(session)
        await run4(session)


if __name__ == "__main__":
    asyncio.run(main())
