# -*- coding: utf-8 -*-
"""P0 结果汇总：扫描 outputs/p0_*.jsonl → records/p0_results.csv（可直接往主表里抄）。

用法（在 TFA_SVA/ 下，或任意目录）：
    python summarize_results.py                       # 默认扫 ../outputs
    python summarize_results.py --dir ../outputs --out ../records/p0_results.csv

文件命名约定（由 run_p0.sh 生成）：
    p0_A_{method}_{dataset}.jsonl              Setting-A（3 指纹模型）
    p0_B{if|hash|imf}_{method}_{dataset}.jsonl Setting-B（1 指纹 + 2×base）
其中 dataset ∈ {if, hash, imf, arc, gsm}。

判分规则与仓库既有后处理完全一致：
    if/hash/imf → fingerprint_parse_pred_ans（label 出现在 pred 或 pred_solution 即命中）
    arc         → arc_parse_pred_ans（pred 中第一个字母 vs label）
    gsm         → gsm_parse_pred_ans（pred 与 label 数值相等）→ 指标为 ACC
已解析过的文件首行是 {"accuracy": ...}，脚本优先直接读它；否则现场重算。
"""
import argparse
import csv
import json
import re
from pathlib import Path

FP_SETS = {"if": "IF", "hash": "Hash", "imf": "ImF"}


def load_items(path):
    """返回 (items, acc_from_first_line)。首行的 {"accuracy": ...} 由后处理写入。"""
    items, acc = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                jo = json.loads(line)
            except Exception:
                continue
            if list(jo.keys()) == ["accuracy"]:
                acc = float(jo["accuracy"])
                continue
            items.append(jo)
    return items, acc


def compute(items, dataset):
    """重算指标（与 utils/ans_process.py 一致）。返回 (value, n)。"""
    seen, correct, total = set(), 0, 0
    for jo in items:
        if dataset in FP_SETS:
            q = jo.get("question")
            if q in seen:
                continue
            seen.add(q)
            total += 1
            label = str(jo.get("label", "") or "").strip()
            pred = str(jo.get("pred", "") or "")
            psol = str(jo.get("pred_solution", "") or "")
            if label and (label in pred or label in psol):
                correct += 1
        elif dataset == "arc":
            q = jo.get("question")
            if q in seen:
                continue
            seen.add(q)
            total += 1
            letter = next((c.upper() for c in str(jo.get("pred", "")) if c.isalpha()), " ")
            if letter == str(jo.get("label", "")).strip().upper():
                correct += 1
        else:  # gsm
            key = jo.get("original_sln") or jo.get("question")
            if key in seen:
                continue
            seen.add(key)
            total += 1
            pred, label = jo.get("pred"), jo.get("label")
            if isinstance(label, str):
                m = re.search(r"#### (-?\d+)", label)
                label = float(m.group(1)) if m else None
            if pred is not None and label is not None and pred == label:
                correct += 1
    return (float(correct) / total if total else float("nan")), total


def parse_name(stem):
    """p0_A_vanilla_if / p0_Bhash_ours_gsm → (scenario, group, method, dataset)"""
    parts = stem.split("_")
    if len(parts) != 4 or parts[0] != "p0":
        return None
    tag, method, dataset = parts[1], parts[2], parts[3]
    if dataset not in ("if", "hash", "imf", "arc", "gsm"):
        return None
    if tag == "A":
        return "Setting-A(3fp)", "-", method, dataset
    if tag.startswith("B") and tag[1:] in FP_SETS:
        return "Setting-B(1fp)", tag[1:], method, dataset
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "outputs"))
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "records" / "p0_results.csv"))
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("p0_*.jsonl"))
    if not files:
        print("未找到 %s/p0_*.jsonl —— 先跑 run_p0.sh" % args.dir)
        return

    rows = []
    for p in files:
        parsed = parse_name(p.stem)
        if not parsed:
            print("跳过（命名不符）：%s" % p.name)
            continue
        scenario, group, method, dataset = parsed
        items, acc = load_items(p)
        if acc is None:
            value, n = compute(items, dataset)
            src = "recomputed"
        else:
            value, n = acc, len(items) or 0
            src = "first-line"
        metric = "fsr" if dataset in FP_SETS else "acc"
        fp = FP_SETS.get(dataset) or ("mixed(3fp)" if group == "-" else FP_SETS[group])
        rows.append(dict(scenario=scenario, group=group, method=method, dataset=dataset,
                         fingerprint=fp, metric=metric, value=value, n=n,
                         source=src, file=p.name))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "group", "method", "dataset", "fingerprint", "metric",
                    "value", "n", "source", "file"])
        for r in rows:
            w.writerow([r["scenario"], r["group"], r["method"], r["dataset"], r["fingerprint"],
                        r["metric"], "%.4f" % r["value"], r["n"], r["source"], r["file"]])

    print("=== 汇总（%d 个文件）→ %s ===" % (len(files), out))
    print("%-16s %-6s %-11s %-8s %-6s %7s %5s" %
          ("scenario", "group", "method", "dataset", "metric", "value", "n"))
    for r in sorted(rows, key=lambda x: (x["scenario"], x["group"], x["method"], x["dataset"])):
        print("%-16s %-6s %-11s %-8s %-6s %7.4f %5d" %
              (r["scenario"], r["group"], r["method"], r["dataset"], r["metric"], r["value"], r["n"]))

    # 覆盖度自检：Setting-A 应有 4 方法 × 5 数据集；Setting-B 应有 3 组 × 4 方法 × 5 数据集
    have = {(r["scenario"], r["group"], r["method"], r["dataset"]) for r in rows}
    methods = ("vanilla", "median", "ours", "thresh_ours")
    datasets = ("if", "hash", "imf", "arc", "gsm")
    todo = [("Setting-A(3fp)", "-", m, d) for m in methods for d in datasets
            if ("Setting-A(3fp)", "-", m, d) not in have]
    todo += [("Setting-B(1fp)", g, m, d) for g in FP_SETS for m in methods for d in datasets
             if ("Setting-B(1fp)", g, m, d) not in have]
    if todo:
        print("\n还缺的实验（%d 个）：" % len(todo))
        for x in todo:
            print("  %s | group=%s | %-11s | %s" % x)


if __name__ == "__main__":
    main()

