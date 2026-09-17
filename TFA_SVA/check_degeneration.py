# -*- coding: utf-8 -*-
"""生成健康度诊断：判断某个方法/温度档的输出是否"退化（乱码/复读/崩坏）"。

动机：`temperature` 的 T 扫描中，**FSR/ACC 不足以暴露退化** ——
  · ARC 的答案只是一个字母（A–D），即使分布已被抹平也能吐对；
  · 乱码/退化最可能出现在**长生成**（ImF 整句改写）与**非 ASCII 答案**（IF 的日语指纹）上。
因此定档前用本脚本对比"候选档 vs 参考档（通常 T=1）"的健康度指标。

用法（在 TFA_SVA/ 下）：
    python check_degeneration.py ../outputs/probe_temp_T1.5_arc10.jsonl
    python check_degeneration.py ../outputs/probe_temp_T*.jsonl ../outputs/p0_A_temperature_arc.jsonl

指标（逐文件）：
    n           有效条目数（自动跳过首行 {"accuracy": ...}）
    ACC         首行 accuracy（若有）
    规范率      首字符 ∈ {A,B,C,D}（ARC 口径；其它任务看 0/1 是否合理）
    长度(均/中) 生成文本字符数
    非ASCII     非 ASCII 字符占比（IF 的日语答案会抬高，正常）
    4-gram 去重 去重 4-gram / 总 4-gram（**越低越像复读/退化**；<0.8 视为可疑）
    distinct    去重 token / 总 token
最后打印每个文件的 2 条样例，供人眼复核。
"""
import argparse
import json
import re
from pathlib import Path


def load_rows(path):
    rows, acc = [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")          # 容忍 BOM
            if not line:
                continue
            try:
                jo = json.loads(line)
            except Exception:
                continue
            if list(jo.keys()) == ["accuracy"]:
                acc = jo["accuracy"]
                continue
            rows.append(jo)
    return rows, acc


def ngrams(toks, k=4):
    return [tuple(toks[i:i + k]) for i in range(max(0, len(toks) - k + 1))]


def max_run(text):
    """最长"同一字符连续重复"长度（`0!!!!!!!` 这类退化输出会给出 20+）。"""
    best = run = 0
    prev = None
    for c in text:
        run = run + 1 if c == prev else 1
        prev = c
        best = max(best, run)
    return best


def stats(path):
    rows, acc = load_rows(path)
    texts = [str(r.get("pred_solution") or r.get("pred") or "") for r in rows]
    n = len(texts)
    if not n:
        return None
    ok = sum(1 for t in texts if t.strip()[:1].upper() in list("ABCD"))
    lens = [len(t) for t in texts]
    nonascii = sum(sum(1 for c in t if ord(c) > 127) for t in texts) / max(1, sum(lens))
    tok_lists = [re.findall(r"\w+|[^\w\s]", t) for t in texts]
    tot_tok = sum(len(x) for x in tok_lists)
    distinct = sum(len(set(x)) for x in tok_lists) / max(1, tot_tok)
    g, gu = 0, 0
    for x in tok_lists:
        gs = ngrams(x)
        g += len(gs)
        gu += len(set(gs))
    gram_ratio = (gu / g) if g >= 20 else None      # 4-gram 太少（短答案）时不作判据
    runs = [max_run(t) for t in texts]
    return dict(n=n, acc=acc, well=ok, mean_len=sum(lens) / n,
                med_len=sorted(lens)[n // 2], nonascii=nonascii,
                distinct=distinct, gram_ratio=gram_ratio, n_grams=g,
                max_run=max(runs) if runs else 0,
                frac_run8=sum(1 for r in runs if r >= 8) / n,   # 重复符号跑 ≥8 的样本比例
                empties=sum(1 for t in texts if not t.strip()), texts=texts)


def verdict(s, ref):
    """与参考档对比 → (判定字符串, 触发的理由列表)。

    退化判据（相对参考档）：4-gram 去重 ↓>0.15 · distinct ↓>0.15 · 非ASCII ↑>0.10 ·
    空输出比例 ↑>0.20（且参考档 <0.10）· 规范率 ↓>0.20（仅当参考档 ≥0.5，即 ARC 口径）。
    """
    if ref is None:
        return "-", []
    reasons = []

    def g(k):
        return s.get(k), ref.get(k)
    a, b = g("gram_ratio")
    if a is not None and b is not None and a < b - 0.15:
        reasons.append("4-gram 去重 %.3f→%.3f（复读）" % (b, a))
    a, b = g("distinct")
    if a < b - 0.15:
        reasons.append("distinct %.3f→%.3f" % (b, a))
    a, b = g("nonascii")
    if a > b + 0.10:
        reasons.append("非ASCII %.3f→%.3f（乱码）" % (b, a))
    er, es = ref["empties"] / max(1, ref["n"]), s["empties"] / max(1, s["n"])
    if es > 0.20 and er < 0.10:
        reasons.append("空输出 %.2f→%.2f" % (er, es))
    wr, ws = ref["well"] / max(1, ref["n"]), s["well"] / max(1, s["n"])
    if wr >= 0.5 and ws < wr - 0.20:
        reasons.append("规范率 %.2f→%.2f" % (wr, ws))
    if s["frac_run8"] > ref["frac_run8"] + 0.20 or (s["max_run"] >= 20 and ref["max_run"] < 10):
        reasons.append("重复符号跑长 max_run %d→%d（复读/发散，如 `0!!!!`）"
                       % (ref["max_run"], s["max_run"]))
    return ("⚠️ 疑似退化", reasons) if reasons else ("✅ 与参考档相当", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--ref", type=str, default=None,
                    help="参考档文件（通常是 T=1）；给出后自动对每个文件做退化判定")
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()

    ref = None
    if args.ref:
        ref = stats(Path(args.ref))
        if ref is None:
            print("⚠️ 参考档无有效条目：%s" % args.ref)
        else:
            print("参考档：%s（ACC=%s 规范 %d/%d 4-gram=%s 非ASCII=%.3f distinct=%.3f）\n"
                  % (Path(args.ref).name, ref["acc"], ref["well"], ref["n"],
                     "-" if ref["gram_ratio"] is None else "%.3f" % ref["gram_ratio"],
                     ref["nonascii"], ref["distinct"]))

    print("%-34s %4s %6s %8s %9s %9s %9s %10s %8s  %s" %
          ("file", "n", "ACC", "规范率", "长度均/中", "非ASCII", "distinct", "4-gram去重",
           "max_run", "判定"))
    for p in args.files:
        files = sorted(Path().glob(p)) if any(c in p for c in "*?[") else [Path(p)]
        for f in files:
            s = stats(f)
            if s is None:
                print("%-34s  (无有效条目)" % f.name)
                continue
            gram = "-" if s["gram_ratio"] is None else "%.3f" % s["gram_ratio"]
            tag, reasons = verdict(s, ref)
            print("%-34s %4d %6s %8s %9s %9.3f %9.3f %10s %8d  %s" %
                  (f.name, s["n"], ("%.2f" % s["acc"]) if s["acc"] is not None else "-",
                   "%d/%d" % (s["well"], s["n"]),
                   "%.1f/%.0f" % (s["mean_len"], s["med_len"]),
                   s["nonascii"], s["distinct"], gram, s["max_run"], tag))
            for r in reasons:
                print("%-34s      ↳ %s" % ("", r))
    print("\n样例（人眼复核）：")
    for p in args.files:
        files = sorted(Path().glob(p)) if any(c in p for c in "*?[") else [Path(p)]
        for f in files:
            s = stats(f)
            if s is None:
                continue
            # 优先展示"重复符号跑长"的坏例（如 0!!!!!!），否则展示前几条
            bad = [t for t in s["texts"] if max_run(t) >= 8]
            shown = (bad[:args.samples] + s["texts"][:args.samples])[:args.samples + 1]
            for t in shown:
                mark = "⚠️" if max_run(t) >= 8 else "  "
                print("  %s[%s] %s" % (mark, f.stem[:18], repr(t[:90])))
    if ref is None:
        print("\n提示：加 `--ref <参考档文件>`（通常是 T=1 的输出）可自动给出退化判定；"
              "本次未给参考档，请自行对照上表。")


if __name__ == "__main__":
    main()
