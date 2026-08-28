# -*- coding: utf-8 -*-
"""准备 Utility 评测集：GSM8K + ARC-Challenge，各取前 N 条，转成仓库 collate 需要的格式。

用法（在 TFA_SVA/ 目录下）：
    python prepare_utility_data.py --num 100

输出（默认）：
    datasets/utility/gsm8k_100.jsonl   # 每行 {"instruction","input","output"}
    datasets/utility/arc_100.jsonl     # 每行 {"question","A","B","C","D","answer"}

路径含 "gsm"/"arc" 关键字，脚本的 collate 分派（if 'gsm' in test_set ...）才能识别。
"""
import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def dump_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=100, help="每个数据集取前 N 条")
    parser.add_argument("--out", type=str, default=str(REPO_ROOT / "datasets" / "utility"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset

    # ---- GSM8K ----
    gsm = load_dataset("gsm8k", "main")["test"]
    gsm_items = []
    for q in gsm.select(range(min(args.num, len(gsm)))):
        gsm_items.append({
            "instruction": "Solve the following math problem step by step.",
            "input": q["question"],
            "output": q["answer"],
        })
    dump_jsonl(gsm_items, os.path.join(args.out, f"gsm8k_{len(gsm_items)}.jsonl"))

    # ---- ARC-Challenge ----
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    letters = ["A", "B", "C", "D", "E"]
    arc_items = []
    for q in arc.select(range(min(args.num, len(arc)))):
        item = {"question": q["question"]}
        choices = q["choices"]["text"]
        for i, t in enumerate(choices):
            if i < len(letters):
                item[letters[i]] = t
        item["answer"] = q["answerKey"]
        arc_items.append(item)
    dump_jsonl(arc_items, os.path.join(args.out, f"arc_{len(arc_items)}.jsonl"))


if __name__ == "__main__":
    main()
