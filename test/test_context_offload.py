import requests
import re

# API endpoint
BASE_URL = "http://0.0.0.0:8002/"


def find_offloaded_files(store_dir: str = "/workspace/context_store", pattern: str = "stored in") -> list:
    """
    Find all offloaded file references in the context store

    Args:
        store_dir: Directory where offloaded content is stored
        pattern: Search pattern for file references

    Returns:
        List of file paths found
    """
    response = requests.post(
        url=f"{BASE_URL}grep",
        json={
            "pattern": pattern,
            "path": store_dir,
            "glob": "*.txt",
            "limit": 100
        }
    )

    result = response.json()
    return result


def reload_offloaded_content(file_path: str, offset: int = None, limit: int = None) -> dict:
    """
    Reload full content from an offloaded file

    Args:
        file_path: Absolute path to the offloaded file
        offset: Optional line offset for pagination
        limit: Optional line limit for pagination

    Returns:
        Full content from the file
    """
    params = {"absolute_path": file_path}
    if offset is not None:
        params["offset"] = offset
    if limit is not None:
        params["limit"] = limit

    response = requests.post(
        url=f"{BASE_URL}read_file",
        json=params
    )
    return response.json()



# Example Usage
if __name__ == "__main__":
    # 测试find_offloaded_files
    store_dir = "./cache_file/"
    print(find_offloaded_files(store_dir, pattern = "tool"))

    # load_offloaded_content
    file_path = "./cache_file/16995f21aab44f26ad55e3cd6068e6eb.txt"
    print(reload_offloaded_content(file_path))