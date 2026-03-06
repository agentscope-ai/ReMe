import argparse
import json
import random


def split_jsonl(input_file, train_file, val_file, ratio=0.8):
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    random.shuffle(data)

    split_idx = int(len(data) * ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSONL file into train and validation sets.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file")
    parser.add_argument("--train", required=True, help="Path to output train file")
    parser.add_argument("--val", required=True, help="Path to output validation file")
    parser.add_argument("--ratio", type=float, default=0.5, help="Train ratio (default: 0.8)")

    args = parser.parse_args()
    split_jsonl(args.input, args.train, args.val, args.ratio)
