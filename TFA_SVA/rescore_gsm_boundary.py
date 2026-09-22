# -*- coding: utf-8 -*-
"""GSM8K 评分口径复核：**原口径（取全文最后一个数字）** vs **边界截断口径**。

起因（2026-09-22）：`ctrl_base`（= `base` 单独的贪心解）原口径只有 0.17，但窥视生成发现
   它**先正确写出 Step 1/2/3，然后续写"下一道题"**（`### human: ...`），而 `gsm_extract_math_answer`
   **取全文最后一个数字** ⇒ 取到了幻觉题的答案 ⇒ 判错。按"下一题边界"截断后 base = **0.57**。
   ⇒ 需要把**所有** GSM8K 结果用统一口径重算，才能判断哪些结论受此影响。

做法（只读 outputs/，不跑模型）：
    · 对每个 jsonl 的每条记录，取 `pred_solution`：
        pred_orig  = 原口径（文件里已写好的 `pred` 字段）
        pred_trunc = 先把生成截断到**第一个** `### human/Instruction/Assistant/System:` 之前，再抽数字
    · 统计每个文件的：n / acc_orig / acc_trunc / 多题率(含新边界标记的比例) / nan 数
    · 输出到终端 + `records/gsm_boundary_rescore.csv`

用法（在 TFA_SVA/ 下；纯 CPU，几秒）：
    python rescore_gsm_boundary.py                     # 扫 ../outputs 的 p0_*_gsm.jsonl 与 ctrl_*_gsm.jsonl
    python rescore_gsm_boundary.py --dir ../outputs --out ../records/gsm_boundary_rescore.csv
    python rescore_gsm_boundary.py --only p0_A_vanilla_gsm.jsonl
"""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils.extract_response import gsm_extract_math_answer      # noqa: E402

BOUNDARY = re.compile(r"###\s*(?:human|Instruction|Assistant|System)\s*:", re.I)


def truncate_at_boundary(text):
    """截断到第一个"下一题"边界之前（保留第一段答案的全文）。"""
    return BOUNDARY.split(str(text or ""))[0]


def read_items(path):
    items = []
    acc_first = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if list(d.keys()) == ["accuracy"]:
                acc_first = d["accuracy"]
                continue
            items.append(d)
    return items, acc_first


def is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def score(items):
    """→ dict(acc_orig, acc_trunc, multi, nan_orig, nan_trunc, n)"""
    n = len(items)
    ok_o = ok_t = multi = nan_o = nan_t = 0
    for d in items:
        label, gen = d.get("label"), d.get("pred_solution", "")
        po = d.get("pred")
        pt = gsm_extract_math_answer(truncate_at_boundary(gen))
        if BOUNDARY.search(str(gen)):
            multi += 1
        nan_o += 1 if is_nan(po) else 0
        nan_t += 1 if is_nan(pt) else 0
        ok_o += 1 if (po == label) else 0
        ok_t += 1 if (pt == label) else 0
    return dict(n=n, acc_orig=ok_o / n if n else float("nan"), acc_trunc=ok_t / n if n else float("nan"),
                multi=multi, nan_orig=nan_o, nan_trunc=nan_t, ok_orig=ok_o, ok_trunc=ok_t)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def item_key(d):
    return d.get("original_sln") or d.get("question")


def correct_map(path):
    """→ {item_key: 0/1}（**截断口径**下的对错向量）"""
    items, _ = read_items(path)
    out = {}
    for d in items:
        k = item_key(d)
        if k is None or k in out:
            continue
        out[k] = 1 if gsm_extract_math_answer(truncate_at_boundary(d.get("pred_solution", ""))) == d.get("label") else 0
    return out


def parse_p0_sub(name):
    m = re.match(r"p0_(A|Bif|Bhash|Bimf)_(.+)_gsm\.jsonl$", name)
    return (m.group(1), m.group(2)) if m else None


PAIRS = [("gap_supp", "vanilla"), ("gap_supp", "median"), ("gap_supp", "confidence"), ("gap_supp", "clipping"),
         ("gap_supp", "gap_supp_gtau0a1"), ("gap_supp", "gap_supp_gtau85a1"), ("gap_supp", "gap_supp_gtau95a1"),
         ("gap_supp", "gap_supp_gtau90a2"), ("gap_supp", "temperature_T0.5"), ("gap_supp", "temperature_T0.75"),
         ("gap_supp", "temperature_T1.0"), ("gap_supp", "temperature_T1.25"),
         ("vanilla", "median"), ("temperature_T0.5", "median"), ("gap_supp_gtau0a1", "gap_supp_gtau95a1"),
         # 温和 clipping 族（clip_topk，2026-09-22 立项）——GSM8K 上也要与主对照配齐（缺的会自动跳过）
         ("gap_supp", "clip_topk_ctk50p90b1.0"), ("gap_supp", "clip_topk_ctk50p50b0.5"),
         ("gap_supp_gtau90a2", "clip_topk_ctk50p90b1.0"),
         ("vanilla", "clip_topk_ctk50p90b1.0"), ("vanilla", "clip_topk_ctk50p50b0.5"),
         ("median", "clip_topk_ctk50p90b1.0"), ("median", "clip_topk_ctk50p50b0.5"),
         ("clipping", "clip_topk_ctk50p90b1.0"),
         ("clip_topk_ctk50p50b0.5", "clip_topk_ctk50p90b1.0")]


def paired_report(files, out_path):
    """在同一场景内做**截断口径**的 McNemar 精确配对检验 → CSV。"""
    by_sub = {}
    for p in files:
        sub = parse_p0_sub(p.name)
        if sub:
            by_sub.setdefault(sub[0], {})[sub[1]] = p
    ctrl = {p.name[len("ctrl_"):-len("_gsm.jsonl")]: p for p in files if p.name.startswith("ctrl_")}
    rows = []

    def add(tag, sub, a_name, b_name, pa, pb):
        ca, cb = correct_map(pa), correct_map(pb)
        keys = sorted(set(ca) & set(cb))
        if not keys:
            return
        b_only = sum(1 for k in keys if ca[k] == 0 and cb[k] == 1)
        a_only = sum(1 for k in keys if ca[k] == 1 and cb[k] == 0)
        n = len(keys)
        d = (a_only - b_only) * 100.0 / n
        pv = mcnemar_exact(a_only, b_only)
        rows.append(dict(scenario=sub, metric="trunc", method_a=a_name, method_b=b_name, n_paired=n,
                         a_only_correct=a_only, b_only_correct=b_only, delta_pp="%+.2f" % d,
                         p_mcnemar="%.4f" % pv, sig_5pct="yes" if pv < 0.05 else "no", note=tag))

    for sub, methods in sorted(by_sub.items()):
        for a, b in PAIRS:
            if a in methods and b in methods:
                add("p0", sub, a, b, methods[a], methods[b])
    # 恒等式核对：1fp 的 median 应与 ctrl_base 完全一致（不一致对 = 0）
    if "base" in ctrl:
        for sub, methods in sorted(by_sub.items()):
            if "median" in methods:
                add("identity(median vs 3xbase)", sub, "median", "ctrl_base", methods["median"], ctrl["base"])
    if rows:
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n===== 截断口径下的配对检验（McNemar 精确，同题）=====")
        print("%-18s %-24s %-24s %5s %8s %8s %8s %9s" % ("场景", "A", "B", "n", "a_only", "b_only", "Δ(pp)", "p"))
        for r in rows:
            print("%-18s %-24s %-24s %5d %8d %8d %8s %9s%s" % (
                r["scenario"], r["method_a"], r["method_b"], r["n_paired"], r["a_only_correct"],
                r["b_only_correct"], r["delta_pp"], r["p_mcnemar"], " ★" if r["sig_5pct"] == "yes" else ""))
        print("已写出：%s" % out_path)
    else:
        print("未找到可配对的 p0_*_gsm.jsonl（需要同一场景内至少两个方法）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE.parent / "outputs"))
    ap.add_argument("--out", default=str(HERE.parent / "records" / "gsm_boundary_rescore.csv"))
    ap.add_argument("--only", default=None, help="只看某个文件名（子串匹配）")
    ap.add_argument("--paired", action="store_true",
                    help="额外做**截断口径**的 McNemar 配对检验 → records/gsm_boundary_paired.csv")
    ap.add_argument("--paired-out", default=str(HERE.parent / "records" / "gsm_boundary_paired.csv"))
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(list(d.glob("p0_*_gsm.jsonl")) + list(d.glob("ctrl_*_gsm.jsonl")))
    if args.only:
        files = [f for f in files if args.only in f.name]
    if not files:
        print("未找到 %s 下的 p0_*_gsm.jsonl / ctrl_*_gsm.jsonl" % d)
        return

    rows = []
    for p in files:
        items, acc_first = read_items(p)
        if not items:
            print("跳过空文件：%s" % p.name)
            continue
        s = score(items)
        s["file"] = p.name
        s["acc_first"] = acc_first
        rows.append(s)

    print("===== GSM8K 评分口径复核（原口径 = 全文最后一个数字；截断口径 = 先截到第一个答案段）=====")
    print("%-52s %4s %9s %10s %8s %7s %7s" % ("文件", "n", "原口径", "截断口径", "差", "多题率", "nan"))
    print("-" * 108)
    for s in sorted(rows, key=lambda x: x["file"]):
        print("%-52s %4d %9.4f %10.4f %+8.4f %6d%% %7s" % (
            s["file"], s["n"], s["acc_orig"], s["acc_trunc"],
            s["acc_trunc"] - s["acc_orig"], s["multi"], "%d/%d" % (s["nan_orig"], s["nan_trunc"])))
    print("-" * 108)
    d_orig = sum(s["ok_orig"] for s in rows)
    d_tr = sum(s["ok_trunc"] for s in rows)
    n_tot = sum(s["n"] for s in rows)
    print("合计 %d 个文件 / %d 条：原口径 %d 对（%.4f）→ 截断口径 %d 对（%.4f）" % (
        len(rows), n_tot, d_orig, d_orig / n_tot, d_tr, d_tr / n_tot))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "n", "acc_orig", "acc_trunc", "delta", "multi", "nan_orig", "nan_trunc", "acc_first_line"])
        for s in sorted(rows, key=lambda x: x["file"]):
            w.writerow([s["file"], s["n"], "%.4f" % s["acc_orig"], "%.4f" % s["acc_trunc"],
                        "%+.4f" % (s["acc_trunc"] - s["acc_orig"]), s["multi"],
                        s["nan_orig"], s["nan_trunc"], s["acc_first"]])
    print("已写出：%s" % out)

    if args.paired:
        paired_report(files, args.paired_out)


if __name__ == "__main__":
    main()
