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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE.parent / "outputs"))
    ap.add_argument("--out", default=str(HERE.parent / "records" / "gsm_boundary_rescore.csv"))
    ap.add_argument("--only", default=None, help="只看某个文件名（子串匹配）")
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


if __name__ == "__main__":
    main()
