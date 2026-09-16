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
    return dict(n=n, acc=acc, well=ok, mean_len=sum(lens) / n,
                med_len=sorted(lens)[n // 2], nonascii=nonascii,
                distinct=distinct, gram_ratio=gram_ratio, n_grams=g,
                empties=sum(1 for t in texts if not t.strip()), texts=texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--samples", type=int, default=2)
    args = ap.parse_args()

    print("%-34s %4s %6s %8s %9s %9s %9s %10s" %
          ("file", "n", "ACC", "规范率", "长度均/中", "非ASCII", "distinct", "4-gram去重"))
    for p in args.files:
        for f in sorted(Path().glob(p)) if any(c in p for c in "*?[") else [Path(p)]:
            s = stats(f)
            if s is None:
                print("%-34s  (无有效条目)" % f.name)
                continue
            gram = "-" if s["gram_ratio"] is None else "%.3f" % s["gram_ratio"]
            flag = " ⚠️可疑" if (s["gram_ratio"] is not None and s["gram_ratio"] < 0.8) else ""
            print("%-34s %4d %6s %8s %9s %9.3f %9.3f %10s%s" %
                  (f.name, s["n"], ("%.2f" % s["acc"]) if s["acc"] is not None else "-",
                   "%d/%d" % (s["well"], s["n"]),
                   "%.1f/%.0f" % (s["mean_len"], s["med_len"]),
                   s["nonascii"], s["distinct"], gram, flag))
    print("\n样例（人眼复核）：")
    for p in args.files:
        for f in sorted(Path().glob(p)) if any(c in p for c in "*?[") else [Path(p)]:
            s = stats(f)
            if s is None:
                continue
            for t in s["texts"][:args.samples]:
                print("  [%s] %s" % (f.stem[:18], repr(t[:90])))
    print("\n判读：与参考档（T=1）比 —— 4-gram 去重明显下降 / 出现大量非 ASCII 乱码 / 空输出增多 ⇒ 该档已退化，不可作主表口径。")


if __name__ == "__main__":
    main()
