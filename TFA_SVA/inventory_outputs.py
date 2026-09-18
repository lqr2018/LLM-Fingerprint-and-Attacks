# -*- coding: utf-8 -*-
"""outputs/ 覆盖与体检清单：**只读**扫描所有结果 jsonl，回答"我现在有什么、还缺什么、有没有被覆盖"。

动机（2026-09-18）：要连跑多批实验（τ 敏感性 24 次 + GSM8K 6 次 + 可选档），
   文件会涨到 40+；而"重名 ⇒ 悄悄覆盖已冻结的 n=300 结果"是这条流水线上最危险的失误。
   本脚本把每个文件解析成 (场景, 组, 方法, 数据集) + 指标值 + 健康度，并**主动报重复与空文件**。

用法（在 TFA_SVA/ 下，或任意目录）：
    python inventory_outputs.py                       # 扫 ../outputs
    python inventory_outputs.py --dir ../outputs --cap 256
    python inventory_outputs.py --only gsm            # 只看某个数据集（if/hash/imf/arc/gsm）
    python inventory_outputs.py --dups                # 只报"重复覆盖"与"空/无法解析"的文件

判读要点：
  · **同一 (场景,组,数据集,方法) 出现 2 次** ⇒ 有一次跑了同名文件（后跑的覆盖先跑的）⇒ 先查 n 与 mtime。
  · **n 不一致**（如 100 vs 300）而方法名相同 ⇒ 口径混用风险，主表必须统一 n。
  · **空文件 / 0 条** ⇒ 崩溃或未完成（会污染 summarize 的 nan 行）。
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import summarize_results as SR          # noqa: E402  （复用其命名解析与判分口径）
from check_gsm_output import analyze    # noqa: E402  （复用其退化/截断判定）


def parse_any(stem, legacy=False):
    if stem.startswith(("mdg_", "gsp_")):
        return SR.parse_mdg_name(stem)
    p = SR.parse_name(stem)
    if p is None and legacy:
        p = SR.parse_legacy_name(stem)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE.parent / "outputs"))
    ap.add_argument("--cap", type=int, default=256, help="截断判定用的 max_new_tokens（gsm 用）")
    ap.add_argument("--only", default=None, help="只看某数据集：if/hash/imf/arc/gsm")
    ap.add_argument("--dups", action="store_true", help="只报重复/空/无法解析")
    ap.add_argument("--legacy", action="store_true", help="额外尝试解析旧的 ens* 命名")
    args = ap.parse_args()

    d = Path(args.dir)
    files = sorted(d.glob("*.jsonl"))
    if not files:
        print("未找到 %s/*.jsonl" % d)
        return

    rows, skipped, empty = [], [], []
    for p in files:
        parsed = parse_any(p.stem, args.legacy)
        try:
            items, acc_first = SR.load_items(str(p))
        except Exception as e:
            print("读取失败：%s（%s）" % (p.name, e))
            continue
        if not items and acc_first is None:
            empty.append(p.name)
            continue
        if parsed is None:
            skipped.append((p.name, len(items)))
            continue
        scenario, group, method, dataset = parsed
        if args.only and dataset != args.only:
            continue
        value, n = SR.compute(items, dataset)
        if acc_first is not None:
            value, n = acc_first, (n or len(items))
        health = "-"
        if dataset == "gsm":
            nan = degen = nd = trunc = 0
            for jo in items:
                flags = analyze(jo.get("pred_solution", ""), args.cap)[1]
                nan += isinstance(jo.get("pred"), float) and jo.get("pred") != jo.get("pred")
                degen += "疑似复读" in flags
                nd += "无数字" in flags
                trunc += "疑似截断" in flags
            health = "nan=%d/无数字=%d/复读=%d/截断=%d" % (nan, nd, degen, trunc)
        rows.append(dict(scenario=scenario, group=group, method=method, dataset=dataset,
                         value=value, n=n, health=health, file=p.name,
                         mtime=p.stat().st_mtime, size=p.stat().st_size))

    if not args.dups:
        print("===== 覆盖与体检清单（%s，共 %d 个可解析文件）=====" % (d, len(rows)))
        hdr = "%-16s %-5s %-34s %-5s %5s %8s  %-30s %s" % (
            "场景", "组", "方法", "数据集", "n", "指标值", "健康(仅 gsm)", "文件")
        print(hdr)
        print("-" * len(hdr))
        for r in sorted(rows, key=lambda x: (x["scenario"], x["group"], x["method"], x["dataset"])):
            print("%-16s %-5s %-34s %-5s %5d %8.4f  %-30s %s" % (
                r["scenario"], r["group"], r["method"], r["dataset"], r["n"], r["value"],
                r["health"], r["file"]))

    key = lambda r: (r["scenario"], r["group"], r["dataset"], r["method"])
    buckets = defaultdict(list)
    for r in rows:
        buckets[key(r)].append(r)
    dups = {k: v for k, v in buckets.items() if len(v) > 1}

    # 口径混用检查（更常见、更危险）：同 (场景,组,数据集) 下方法名互为前缀但 n 不同
    #   例：gap_supp_gtau90a1 (n=100) 与 gap_supp_gtau90a1_arc300 (n=300) ⇒ 写表时千万别混用
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["scenario"], r["group"], r["dataset"])].append(r)
    mixes = []
    for k, v in by_cell.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                a, b = v[i], v[j]
                if a["method"] == b["method"] or a["n"] == b["n"]:
                    continue
                if b["method"].startswith(a["method"]) or a["method"].startswith(b["method"]):
                    mixes.append((k, a, b))

    print("")
    print("===== 风险检查 =====")
    if dups:
        print("⚠️ 重复覆盖（同 场景/组/数据集/方法 有多份 ⇒ 后跑的同名文件会覆盖先跑的）：")
        for k, v in dups.items():
            ns = sorted({x["n"] for x in v})
            print("  %s | n=%s" % (" / ".join(k), ns))
            for x in sorted(v, key=lambda y: y["mtime"]):
                print("      %s  n=%d  值=%.4f  %s" % (x["file"], x["n"], x["value"],
                                                       __import__("time").strftime("%Y-%m-%d %H:%M", __import__("time").localtime(x["mtime"]))))
    if not dups:
        print("✅ 无重复覆盖（每个 (场景,组,数据集,方法) 只有一份）")
    if mixes:
        print("⚠️ 同配置存在不同 n（口径混用风险 —— 写表/做检验时只能用同 n 的那批）：")
        for (k, a, b) in mixes:
            print("  %s | %s (n=%d, %.4f ← %s)  vs  %s (n=%d, %.4f ← %s)"
                  % (" / ".join(k), a["method"], a["n"], a["value"], a["file"],
                     b["method"], b["n"], b["value"], b["file"]))
    if empty:
        print("⚠️ 空文件/无内容（会让 summarize 出 nan 行，建议 `bash run_gsm.sh clean` 或手工删）：%s" % ", ".join(empty))
    if skipped:
        print("ℹ️ 命名不符合解析规则（不进 p0_results.csv，属正常：smoke/probe/dbg 等）：")
        for name, k in skipped:
            print("      %s（%d 条）" % (name, k))
    print("")
    print("合计：可解析 %d 份 | 空 %d | 不参与汇总 %d" % (len(rows), len(empty), len(skipped)))


if __name__ == "__main__":
    main()
