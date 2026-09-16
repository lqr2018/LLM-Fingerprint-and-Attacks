# -*- coding: utf-8 -*-
"""P1-3b 真跑结果报表：把 `records/mdg_solo_results*.csv` 汇总成"同批可比"的表。

为什么需要它：
  本次 `mdg_*` 跑的是 ARC-100（n=100），而 P0 基线（vanilla/median/ours/thresh_ours）是 ARC-300（n=300）
  ⇒ 两边的原始 ACC 不能直接比。但配对表里 `mdg_*` 与 P0 基线的 `n_paired` 恰为 100（重叠子集），
  且 `thresh_ours`(n=300) 与 `thresh_ours_p90a1_old`(n=100) 在该子集上**逐题完全相同**（0:0），
  因此可用 Δ 反推基线在**同一批 100 题**上的 ACC：
        acc_base(100) = acc_mdg(100) ± Δ/100     （Δ = a_only − b_only，单位 pp）

用法（仓库根目录）：python TFA_SVA/report_mdg_solo.py
"""
import argparse
import csv

BASELINES = ("vanilla", "median", "ours", "thresh_ours")     # P0：n=300
GATES = ["maxdelta_gate_p90a1", "maxdelta_gate_p90a2", "maxdelta_gate_p95a2",
         "maxdelta_gate_solop85a2", "maxdelta_gate_solop90a2"]
SHORT = {"maxdelta_gate_p90a1": "loo_p90a1", "maxdelta_gate_p90a2": "loo_p90a2",
         "maxdelta_gate_p95a2": "loo_p95a2", "maxdelta_gate_solop85a2": "solo_p85a2",
         "maxdelta_gate_solop90a2": "solo_p90a2"}
ACC_ROWS = [("Setting-B(1fp)", "hash"), ("Setting-B(1fp)", "if"), ("Setting-B(1fp)", "imf"),
            ("Setting-A(3fp)", "-")]
FSR_ROWS = [("Setting-B(1fp)", "hash", "hash"), ("Setting-B(1fp)", "if", "if"),
            ("Setting-B(1fp)", "imf", "imf"), ("Setting-A(3fp)", "-", "if"),
            ("Setting-A(3fp)", "-", "hash"), ("Setting-A(3fp)", "-", "imf")]
SETNAME = {"Setting-B(1fp)": "1fp", "Setting-A(3fp)": "3fp"}


def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=r"records\mdg_solo_results.csv")
    ap.add_argument("--paired", default=r"records\mdg_solo_results_paired.csv")
    args = ap.parse_args()
    res, pairs = load(args.csv), load(args.paired)

    acc = {}
    for r in res:
        if r["value"] in ("nan", "") or r["source"] == "nan":
            continue
        acc[(r["scenario"], r["group"], r["dataset"], r["method"])] = float(r["value"])
    delta = {(r["scenario"], r["group"], r["dataset"], r["method_a"], r["method_b"]):
             (float(r["delta_pp"]), int(r["n_paired"]), float(r["p_mcnemar"])) for r in pairs}

    def base_on_100(sc, grp, ds, name, ref):
        d = delta.get((sc, grp, ds, name, ref))
        if d:
            return acc.get((sc, grp, ds, ref), float("nan")) + d[0] / 100.0
        d = delta.get((sc, grp, ds, ref, name))
        if d:
            return acc.get((sc, grp, ds, ref), float("nan")) - d[0] / 100.0
        return float("nan")

    print("=" * 116)
    print("表1  ACC（ARC）| `mdg_*` = 本次 n=100；基线列 = 用配对行反推的**同一批 100 题**（P0 的 n=300 见记录表）")
    print("=" * 116)
    print("%-14s" % "场景" + "".join(" %10s" % x for x in
                                     ["vanilla", "median", "ours", "thresh"] + [SHORT[m] for m in GATES]))
    for sc, grp in ACC_ROWS:
        row = "%-14s" % ("%s-%s" % (SETNAME[sc], grp) if grp != "-" else SETNAME[sc])
        ref = next((m for m in GATES if (sc, grp, "arc", m) in acc), None)
        for b in BASELINES:
            row += " %10.4f" % base_on_100(sc, grp, "arc", b, ref)
        for m in GATES:
            row += " %10.4f" % acc.get((sc, grp, "arc", m), float("nan"))
        print(row)
    print("\n反推公式：acc_base(100) = acc_mdg(100) ± Δ/100")
    print("内部校验：thresh_ours(n=300) 与 thresh_ours_p90a1_old(n=100) 在 100 道重叠题上 0:0（逐题完全相同）")

    print("\n" + "=" * 116)
    print("表2  FSR（指纹测试集，n=10）")
    print("=" * 116)
    print("%-14s" % "场景-靶子" + "".join(" %10s" % SHORT[m] for m in GATES) + " %10s" % "thresh_old")
    for sc, grp, ds in FSR_ROWS:
        row = "%-14s" % ("%s-%s" % (SETNAME[sc], ds))
        for m in GATES:
            row += " %10.2f" % acc.get((sc, grp, ds, m), float("nan"))
        row += " %10.2f" % acc.get((sc, grp, ds, "thresh_ours_p90a1_old"), float("nan"))
        print(row)

    print("\n" + "=" * 116)
    print("表3  关键配对检验（n_paired=100；Δ>0 表示 a 更好）—— 显著者 + 全部涉及 solo 的比较")
    print("=" * 116)
    names = tuple(BASELINES) + tuple(GATES) + ("thresh_ours_p90a1_old",)
    rows = [r for r in pairs if int(r["n_paired"]) >= 100
            and r["method_a"] in names and r["method_b"] in names]
    for r in sorted(rows, key=lambda x: float(x["p_mcnemar"])):
        p = float(r["p_mcnemar"])
        if not (p < 0.05 or "solop85a2" in (r["method_a"] + r["method_b"])):
            continue
        print("%-10s %-5s %-24s vs %-24s n=%3s %+8s p=%-7s %s" %
              ("%s-%s" % (SETNAME[r["scenario"]], "3fp" if r["group"] == "-" else r["group"]),
               r["dataset"], r["method_a"], r["method_b"], r["n_paired"], r["delta_pp"],
               r["p_mcnemar"], "★" if p < 0.05 else ""))


if __name__ == "__main__":
    main()
