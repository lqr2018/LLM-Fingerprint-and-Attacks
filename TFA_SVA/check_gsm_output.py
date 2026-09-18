# -*- coding: utf-8 -*-
"""诊断 GSM8K 输出：区分「算对 / 没提取到数字(nan) / 有数字但算错 / 缺字段 / 生成退化(复读·无数字·疑似截断)」。

用法（任意目录下直接运行，自动定位仓库 outputs/ 下的文件）：
    python TFA_SVA/check_gsm_output.py                       # 默认读 outputs/util_gsm.jsonl
    python TFA_SVA/check_gsm_output.py <路径>                 # 显式指定文件
    python TFA_SVA/check_gsm_output.py <路径> --items 10      # ★逐条打印前 10 条（看重放/退化/截断）
    python TFA_SVA/check_gsm_output.py <路径> --cap 256       # 声明 --max_new_tokens，用于"疑似截断"判定
    python TFA_SVA/check_gsm_output.py ../outputs/p0_*_gsm.jsonl   # 多个文件（shell 展开）

判读要点（第一版就写在这里，避免每次现推）：
  · **算对 = 0 且 nan 占多数** ⇒ 模型根本没走到答案（格式/能力问题），不是"算错"。
  · **算对 = 0 但有数字** ⇒ 真算错（能力或截断）。
  · **首行 accuracy 与实测不一致** ⇒ 该文件曾被重复后处理（`gsm_parse_pred_ans` 会重写首行）。
"""
import argparse
import itertools
import json
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEGEN_CHAR_FRAC = 0.15   # 不同字符占比低于此 ⇒ 疑似复读/乱码
DEGEN_RUN = 20           # 同一字符连续 ≥ 此长度 ⇒ 疑似复读
ANSWER_MARK = re.compile(r"####|the answer is|The answer is|boxed")


def analyze(text, cap):
    """→ (字符长度, [疑似问题标签])"""
    t = str(text or "")
    n = len(t)
    if n == 0:
        return n, ["空输出"]
    flags = []
    if not re.search(r"\d", t):
        flags.append("无数字")
    runs = max((len(list(g)) for _, g in itertools.groupby(t)), default=0)
    if runs >= DEGEN_RUN or (n >= 20 and len(set(t)) / float(n) < DEGEN_CHAR_FRAC):
        flags.append("疑似复读")
    if cap and n >= 3 * cap and not ANSWER_MARK.search(t):
        flags.append("疑似截断")
    return n, flags


def one_file(path, items, cap):
    p = Path(path)
    if not p.exists():
        print(f"文件不存在: {path}")
        return
    match = nan = mismatch = missing = 0
    marked = degen = nodigit = trunc = 0
    lens, nan_examples, rows = [], [], []
    acc_first = None
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
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
            pred, label = d.get("pred"), d.get("label")
            gen = d.get("pred_solution", "")
            n, flags = analyze(gen, cap)
            lens.append(n)
            if ANSWER_MARK.search(str(gen)):
                marked += 1
            if "疑似复读" in flags:
                degen += 1
            if "无数字" in flags:
                nodigit += 1
            if "疑似截断" in flags:
                trunc += 1
            if pred is None or label is None:
                missing += 1
                verdict = "缺字段"
            elif isinstance(pred, float) and math.isnan(pred):
                nan += 1
                verdict = "NAN"
                if len(nan_examples) < 3:
                    nan_examples.append(str(gen)[:120])
            elif pred == label:
                match += 1
                verdict = "OK"
            else:
                mismatch += 1
                verdict = "WRONG"
            rows.append((i, n, flags, pred, label, verdict, str(gen)))

    total = match + nan + mismatch + missing
    print(f"===== 文件: {p} =====")
    if acc_first is not None:
        print(f"首行 accuracy（后处理写入）= {acc_first:.4f}")
    print(f"算对: {match} | 没提取到数字(nan): {nan} | 有数字但算错: {mismatch} | 缺字段: {missing}"
          f"   （共 {total} 条）")
    if lens:
        srt = sorted(lens)
        print(f"生成字符长度: 中位 {srt[len(srt) // 2]} / 最短 {srt[0]} / 最长 {srt[-1]}")
    print(f"生成里出现答案标记（####/the answer is/boxed）: {marked}/{len(lens)}"
          f" | 无数字: {nodigit} | 疑似复读: {degen} | 疑似截断: {trunc}"
          f"（--cap={cap}，0 表示不判截断）")
    if items > 0:
        print(f"--- 逐条明细（前 {items} 条）---")
        for (i, n, flags, pred, label, verdict, gen) in rows[:items]:
            head = gen[:110].replace("\n", " ")
            tail = gen[-70:].replace("\n", " ") if len(gen) > 70 else ""
            print(f"  #{i} len={n} {flags if flags else ''} pred={pred!r} label={label!r} → {verdict}")
            print(f"      head: {head!r}")
            if tail:
                print(f"      tail: {tail!r}")
    if nan_examples:
        print("--- nan 样例（模型没输出数字的生成开头）---")
        for e in nan_examples:
            print(repr(e))
    print("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=None,
                    help="结果 jsonl 路径（可多个/通配符）；缺省 = 仓库 outputs/util_gsm.jsonl")
    ap.add_argument("--items", type=int, default=0, help="逐条打印前 N 条（0=关闭）")
    ap.add_argument("--cap", type=int, default=0, help="该次运行的 --max_new_tokens（用于疑似截断判定）")
    args = ap.parse_args()

    paths = args.paths or [str(REPO_ROOT / "outputs" / "util_gsm.jsonl")]
    for path in paths:
        one_file(path, args.items, args.cap)


if __name__ == "__main__":
    main()
