import json

with open("../../file_vector_store/bfcl_test.jsonl", "r") as f:
    bfcl = [json.loads(line) for line in f]

new_bfcl = []
for exp in bfcl:
    new_exp = {}
    new_exp["workspace_id"] = exp["workspace_id"]
    new_exp["memory_id"] = exp["unique_id"]
    new_exp["memory_type"] = exp["metadata"]["memory_type"]

    new_exp["when_to_use"] = exp["content"]
    new_exp["content"] = exp["metadata"]["content"]
    new_exp["score"] = exp["metadata"]["score"]

    new_exp["time_created"] = exp["metadata"]["time_created"]
    new_exp["time_modified"] = exp["metadata"]["time_modified"]
    new_exp["author"] = exp["metadata"]["author"]

    new_exp["metadata"] = exp["metadata"]["metadata"]
    new_exp["metadata"]["utility"] = 0
    new_exp["metadata"]["freq"] = 0

    new_bfcl.append(new_exp)

with open("../../library/bfcl_test.jsonl", "w", encoding="utf-8") as f:
    f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in new_bfcl)
