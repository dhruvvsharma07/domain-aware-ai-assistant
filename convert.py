import json

with open("data.json", "r") as f:
    data = json.load(f)

with open("train.jsonl", "w") as f:
    for item in data:
        q = item["question"]
        a = item["answer"]

        line = {
            "text": f"### Instruction:\n{q}\n\n### Response:\n{a}"
        }

        f.write(json.dumps(line) + "\n")

print("✅ train.jsonl created successfully!")