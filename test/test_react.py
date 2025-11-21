import requests
import json

url = "http://0.0.0.0:8002/react"
headers = {"Content-Type": "application/json"}

payload = {
    "query":"""
    北京时间11月20日晚，美股大跌，发生了什么？
    """
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
