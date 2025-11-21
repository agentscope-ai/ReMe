import asyncio
import json
import aiohttp
import requests

# API base URL (default is http://0.0.0.0:8002)
base_url = "http://0.0.0.0:8002"
workspace_id = "personal_memory_demo"
#
# # Example conversation with personal details
# messages = [
#     {"role": "user", "content": "My name is John Smith, I'm 28 years old"},
#     {"role": "assistant", "content": "Nice to meet you, John!"},
#     {"role": "user", "content": "I'm a software engineer working with Python"},
#     {"role": "assistant", "content": "I see, you're a Python engineer."},
#     # Additional conversation messages...
# ]
#
# response = requests.post(
#     f"{base_url}/summary_personal_memory",
#     json={
#         "trajectories": [
#             {"messages": messages, "score": 1.0}
#         ],
#         "workspace_id": workspace_id,
#     },
#     headers={"Content-Type": "application/json"}
# )
# print(response.json())
#
# # Example queries to retrieve personal information
# queries = [
#     "What's my name and age?",
#     "What do I do for work?",
#     "What are my hobbies?"
# ]
#
# response = requests.post(
#         f"{base_url}/retrieve_personal_memory",
#         json={
#             "query": queries[0],
#             "workspace_id": workspace_id,
#         },
#         headers={"Content-Type": "application/json"}
#     )
# result = response.json()
# print(f"Query: {queries[0]}")
# print(f"Answer: {result.get('answer', '')}")


import requests
import json

url = "http://0.0.0.0:8002/context_offload"
headers = {"Content-Type": "application/json"}

payload = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "11111111"},
        {"role": "assistant", "content": "2222222"},
        {"role": "user", "content": "333333333"},
        {"role": "assistant", "content": "444444444"},
        {"role": "user", "content": "5555 55 "*2},
        {"role": "tool","content": "tool222 "*10},
        {"role": "user", "content": "6666 66 " * 2},
        {"role": "tool", "content": "tool1111 " * 10},
    ],
    "context_manage_mode": "compact",
    # "context_manage_mode": "auto",
    "max_total_tokens": 5,
    "max_tool_message_tokens": 1,
    "keep_recent_count": 0,
    "store_dir": "./cache_file/",
    "chat_id": "research_session_001"
}

# 发送 POST 请求
resp = requests.post(url, headers=headers, data=json.dumps(payload))

# 打印状态码和服务器返回的文本
print(resp.json())
print("Status:", resp.status_code)
print("Response:", resp.text)

# 如果想把返回内容落盘，方便后续查看，可取消下面几行的注释
# with open("response.json", "w", encoding="utf-8") as f:
#     f.write(resp.text)
