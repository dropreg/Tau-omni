# client.py
import requests

SERVER_URL = "http://127.0.0.1:8000"


def sync_list(data):

    resp = requests.post(
        f"{SERVER_URL}/sync",
        json={"data": data}
    )
    resp.raise_for_status()
    return resp.json()

if __name__ == "__main__":
    # 第一次提交
    result = sync_list([1, 2, 3])
    print("Submit result:", result)

    # 第二次提交（不会覆盖）
    result = sync_list([4, 5, 6])
    print("Submit result:", result)
