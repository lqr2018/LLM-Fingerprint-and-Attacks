# -*- coding: utf-8 -*-
"""准备 Utility 评测集：GSM8K + ARC-Challenge，各取前 N 条，转成仓库 collate 需要的格式。

用法（在 TFA_SVA/ 目录下）：
    python prepare_utility_data.py --num 300

输出（默认）：
    datasets/utility/gsm8k_300.jsonl       # 每行 {"instruction","input","output"}
    datasets/utility/arc_300.jsonl         # 每行 {"question","A","B","C","D","answer"}
    datasets/utility/arc_clean_100.jsonl   # Clean 标定集（ARC train 划分），供 thresh_ours 计算 τ

注意：`arc_clean_*` 取自 ARC-Challenge 的 **train** 划分，与 `arc_*`（test 划分）不重叠，
因此 τ 标定不会看到任何测试样本；`thresh_ours` 请用 `--clean_path ../datasets/utility/arc_clean_100.jsonl`。

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
    parser.add_argument("--clean_num", type=int, default=100,
                        help="额外生成 Clean 标定集（ARC train 划分）的条数；0 表示不生成")
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
    arc_items, skipped = [], 0
    for q in arc:
        if len(arc_items) >= args.num:
            break
        choices = q["choices"]["text"]
        # ⚠️ utils/collate_fun.arc_collate_fn 硬取 A/B/C/D，且模板只展示 A~D：
        #    因此只保留恰好 4 个选项的题（3 选项会 KeyError，5 选项会丢掉 E 导致答案无法解析）
        if len(choices) != 4:
            skipped += 1
            continue
        item = {"question": q["question"]}
        for i, t in enumerate(choices):
            item[letters[i]] = t
        item["answer"] = q["answerKey"]
        arc_items.append(item)
    dump_jsonl(arc_items, os.path.join(args.out, f"arc_{len(arc_items)}.jsonl"))
    if skipped:
        print(f"  (ARC test 划分跳过 {skipped} 条非 4 选项题，保证 prompt 固定为 A~D)")

    # ---- ARC-Challenge（train 划分）→ Clean 标定集：供 thresh_ours 计算 τ，与上面的 test 划分不重叠 ----
    if args.clean_num > 0:
        arc_train = load_dataset("allenai/ai2_arc", "ARC-Challenge")["train"]
        clean_items = []
        for q in arc_train:
            if len(clean_items) >= args.clean_num:
                break
            choices = q["choices"]["text"]
            if len(choices) != 4:
                continue
            item = {"question": q["question"]}
            for i, t in enumerate(choices):
                item[letters[i]] = t
            item["answer"] = q["answerKey"]
            clean_items.append(item)
        dump_jsonl(clean_items, os.path.join(args.out, f"arc_clean_{len(clean_items)}.jsonl"))


if __name__ == "__main__":
    main()
