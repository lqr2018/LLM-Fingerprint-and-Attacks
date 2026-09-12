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
import math
import re
from pathlib import Path

FP_SETS = {"if": "IF", "hash": "Hash", "imf": "ImF"}
PAIR_LIST = [("vanilla", "ours"), ("vanilla", "median"), ("ours", "median"),
             ("ours", "thresh_ours"), ("vanilla", "thresh_ours"), ("median", "thresh_ours")]


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


def is_correct(jo, dataset):
    """单条样本是否判对（与 utils/ans_process.py 的判分口径一致）。"""
    if dataset in FP_SETS:
        label = str(jo.get("label", "") or "").strip()
        pred = str(jo.get("pred", "") or "")
        psol = str(jo.get("pred_solution", "") or "")
        return bool(label) and (label in pred or label in psol)
    if dataset == "arc":
        letter = next((c.upper() for c in str(jo.get("pred", "")) if c.isalpha()), " ")
        return letter == str(jo.get("label", "")).strip().upper()
    # gsm：pred 与 label 均取数值
    pred, label = jo.get("pred"), jo.get("label")
    if isinstance(label, str):
        m = re.search(r"#### (-?\d+)", label)
        label = float(m.group(1)) if m else None
    return pred is not None and label is not None and pred == label


def compute(items, dataset):
    """重算指标（与 utils/ans_process.py 一致）。返回 (value, n)。"""
    seen, correct, total = set(), 0, 0
    for jo in items:
        key = (jo.get("original_sln") if dataset == "gsm" else jo.get("question")) \
            or jo.get("question")
        if key in seen:
            continue
        seen.add(key)
        total += 1
        if is_correct(jo, dataset):
            correct += 1
    return (float(correct) / total if total else float("nan")), total


def per_item_correct(path, dataset):
    """返回 {样本键: 0/1}，用于配对检验（键与 compute 的去重口径一致）。"""
    items, _ = load_items(path)
    seen, out = set(), {}
    for jo in items:
        key = (jo.get("original_sln") if dataset == "gsm" else jo.get("question")) \
            or jo.get("question")
        if key is None or key in seen:
            continue
        seen.add(key)
        out[key] = 1 if is_correct(jo, dataset) else 0
    return out


def mcnemar_exact(b, c):
    """McNemar 精确检验（双侧）：b/c 为两种不一致的计数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def parse_name(stem):
    """p0_A_vanilla_arc / p0_Bhash_thresh_ours_gsm
    → (scenario, group, method, dataset)

    注意：method 可能自带下划线（thresh_ours），因此**不能**按固定段数解析：
    约定为 `p0_{tag}_{method}_{dataset}`，dataset 固定取最后一段，tag 取第 2 段，
    中间剩下的全部拼回 method。
    """
    parts = stem.split("_")
    if len(parts) < 4 or parts[0] != "p0":
        return None
    tag, dataset = parts[1], parts[-1]
    method = "_".join(parts[2:-1])
    if not method or dataset not in ("if", "hash", "imf", "arc", "gsm"):
        return None
    if tag == "A":
        return "Setting-A(3fp)", "-", method, dataset
    if tag.startswith("B") and tag[1:] in FP_SETS:
        return "Setting-B(1fp)", tag[1:], method, dataset
    return None


FP_ALIAS = {"if": "if", "hash": "hash", "stego": "imf", "imf": "imf"}


def parse_legacy_name(stem):
    """兼容旧命名（既有 experiments 的原始输出），返回 (scenario, group, method, dataset)。

    仅处理**指纹数据集**；旧 ARC 文件是 n=100 口径，不混入（避免与 P0-2 的 ARC-300 混淆）。
    命名来源：
        Setting-A 主对比 : ens_{method}_{if|hash|stego}          （run_ensemble_hash_stego.sh 等）
        Setting-A 门控   : ens_thresh{85|90|95}_{if|hash|stego}  / ens_thresh_{hash|stego}_t{t}_a{a}
        Setting-B        : ens1fp_{method}_{if|hash|stego}_a{a}  / ens1fp_thresh_{tag}_a{a}
    约定：**τ_pct=90 且 α=1 的门控结果**才记为 `thresh_ours`（与该设置下的既有口径一致），
          其余变体记为 `thresh_ours_p{t}` / `thresh_ours_p{t}_a{a}`，只进结果表、不参与配对比较。
    """
    if not stem.startswith("ens"):
        return None
    m = re.match(r"^ens1fp_thresh_(if|hash|stego|imf)_a[\d.]+$", stem)
    if m:
        ds = FP_ALIAS[m.group(1)]
        return "Setting-B(1fp)", ds, "thresh_ours", ds
    m = re.match(r"^ens1fp_(?P<m>[a-z_]+?)_(if|hash|stego|imf)_a[\d.]+$", stem)
    if m:
        if "arc" in m.group("m"):      # ens1fp_arc_{method}_{tag}_a{α} 是 ARC 文件 → 排除
            return None
        ds = FP_ALIAS[m.group(2)]
        return "Setting-B(1fp)", ds, m.group("m"), ds
    m = re.match(r"^ens_thresh_(hash|stego)_t(?P<t>\d+)_a(?P<a>[\d.]+)$", stem)
    if m:
        ds = FP_ALIAS[m.group(1)]
        meth = "thresh_ours" if (m.group("t") == "90" and float(m.group("a")) == 1.0) \
            else "thresh_ours_p%s_a%s" % (m.group("t"), m.group("a"))
        return "Setting-A(3fp)", "-", meth, ds
    m = re.match(r"^ens_thresh(?P<t>\d+)_(if|hash|stego|imf)$", stem)
    if m:
        ds = FP_ALIAS[m.group(2)]
        meth = "thresh_ours" if m.group("t") == "90" else "thresh_ours_p%s" % m.group("t")
        return "Setting-A(3fp)", "-", meth, ds
    m = re.match(r"^ens_(?P<m>vanilla|median|ours|random|temperature|clipping|confidence)_"
                 r"(if|hash|stego|imf)$", stem)
    if m:
        return "Setting-A(3fp)", "-", m.group("m"), FP_ALIAS[m.group(2)]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "outputs"))
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "records" / "p0_results.csv"))
    ap.add_argument("--paired", action="store_true",
                    help="额外做配对 McNemar 精确检验（读逐题对错，输出 records/p0_paired.csv）")
    ap.add_argument("--legacy", action="store_true",
                    help="改用旧命名（outputs/ens*.jsonl，既有实验的原始输出）→ 无需重跑即可做配对检验")
    args = ap.parse_args()

    pattern = "ens*.jsonl" if args.legacy else "p0_*.jsonl"
    files = sorted(Path(args.dir).glob(pattern))
    if not files:
        print("未找到 %s/%s%s" % (args.dir, pattern,
                                "（旧输出可能已被清理，那就只能跑 run_p0.sh fp）" if args.legacy else " —— 先跑 run_p0.sh"))
        return

    rows, skipped = [], []
    for p in files:
        parsed = parse_legacy_name(p.stem) if args.legacy else parse_name(p.stem)
        if not parsed:
            print("跳过（命名不符）：%s" % p.name)
            skipped.append(p.name)
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

    print("=== 汇总（%d 个文件，解析 %d，跳过 %d）→ %s ===" % (len(files), len(rows), len(skipped), out))
    if skipped:
        print("⚠️ 有 %d 个文件因命名不符未纳入：%s（请检查是否为 run_p0.sh 的命名）" % (len(skipped), "、".join(skipped)))
    print("%-16s %-6s %-11s %-8s %-6s %7s %5s" %
          ("scenario", "group", "method", "dataset", "metric", "value", "n"))
    for r in sorted(rows, key=lambda x: (x["scenario"], x["group"], x["method"], x["dataset"])):
        print("%-16s %-6s %-11s %-8s %-6s %7.4f %5d" %
              (r["scenario"], r["group"], r["method"], r["dataset"], r["metric"], r["value"], r["n"]))

    # 覆盖度自检（仅对 p0 命名有意义）
    if not args.legacy:
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


    # ---------------- 配对检验（McNemar 精确）：同一批题、同一批模型，比未配对检验功效高 ----------------
    if args.paired:
        table = {}   # (scenario, group, dataset) -> {method: {key: 0/1}}
        for p in files:
            parsed = parse_legacy_name(p.stem) if args.legacy else parse_name(p.stem)
            if not parsed:
                continue
            sc, g, method, dataset = parsed
            table.setdefault((sc, g, dataset), {})[method] = per_item_correct(p, dataset)

        prows = []
        for (sc, g, dataset) in sorted(table):
            methods = table[(sc, g, dataset)]
            for a, b in PAIR_LIST:
                if a not in methods or b not in methods:
                    continue
                common = set(methods[a]) & set(methods[b])
                n_pair = len(common)
                if not n_pair:
                    continue
                a_only = sum(1 for k in common if methods[a][k] and not methods[b][k])
                b_only = sum(1 for k in common if not methods[a][k] and methods[b][k])
                prows.append(dict(scenario=sc, group=g, dataset=dataset, a=a, b=b, n=n_pair,
                                  a_only=a_only, b_only=b_only,
                                  delta=100.0 * (a_only - b_only) / n_pair,
                                  p=mcnemar_exact(a_only, b_only)))

        pout = out.with_name("p0_paired.csv")
        with open(pout, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "group", "dataset", "method_a", "method_b", "n_paired",
                        "a_only_correct", "b_only_correct", "delta_pp", "p_mcnemar", "sig_5pct"])
            for r in prows:
                w.writerow([r["scenario"], r["group"], r["dataset"], r["a"], r["b"], r["n"],
                            r["a_only"], r["b_only"], "%+.2f" % r["delta"], "%.4f" % r["p"],
                            "yes" if r["p"] < 0.05 else "no"])

        print("\n=== 配对检验（McNemar 精确，同一批题；Δ>0 表示 a 更好）→ %s ===" % pout)
        print("%-16s %-6s %-8s %-26s %6s %7s %7s %9s %8s" %
              ("scenario", "group", "dataset", "对比(a-b)", "n", "a_only", "b_only", "Δ(pp)", "p"))
        for r in sorted(prows, key=lambda x: x["p"]):
            print("%-16s %-6s %-8s %-26s %6d %7d %7d %+9.2f %8.4f%s" %
                  (r["scenario"], r["group"], r["dataset"], "%s-%s" % (r["a"], r["b"]),
                   r["n"], r["a_only"], r["b_only"], r["delta"], r["p"],
                   "  ★" if r["p"] < 0.05 else ""))
        sig = [r for r in prows if r["p"] < 0.05]
        print("显著（5%%）的比较：%d / %d" % (len(sig), len(prows)))


if __name__ == "__main__":
    main()

