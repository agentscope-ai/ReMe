"""Split the JSONL file into train and validation sets."""

import argparse
import json
import random

from default_ids import DEFAULT_TRAIN_IDS, DEFAULT_VAL_IDS


def split_jsonl(
    input_file: str,
    train_file: str,
    val_file: str,
    ratio: float = 0.75,
    random_split: bool = False,
) -> None:
    """Split the JSONL file into train and validation sets."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if random_split:
        random.shuffle(data)
        split_idx = int(len(data) * ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
    else:
        train_data = []
        val_data = []
        unknown_ids: list[str] = []
        for obj in data:
            if "id" not in obj:
                raise ValueError(f"Missing 'id' field in input file: {input_file}")
            obj_id = str(obj["id"])
            if obj_id in DEFAULT_TRAIN_IDS:
                train_data.append(obj)
            elif obj_id in DEFAULT_VAL_IDS:
                val_data.append(obj)
            else:
                unknown_ids.append(obj_id)

        if len(train_data) + len(val_data) != len(data):
            missing = len(data) - (len(train_data) + len(val_data))
            examples = ", ".join(unknown_ids) if unknown_ids else "(none)"
            raise ValueError(
                f"{missing} samples in {input_file} not found in train_ref/val_ref id sets. Examples: {examples}",
            )

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
    parser.add_argument(
        "--random",
        action="store_true",
        help="Whether to randomly split input into train/val. "
        "If false, split strictly by default train/val id sets (see default_ids.py).",
    )
    args = parser.parse_args()
    split_jsonl(args.input, args.train, args.val, args.ratio, args.random)
