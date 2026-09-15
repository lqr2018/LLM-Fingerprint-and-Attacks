# -*- coding: utf-8 -*-
"""P1-3 离线筛选：在**已记录的一条轨迹**上比较不同门控方案（决策级指标）。

输入：`output/dbg_imf_ours.jsonl.topn.jsonl`（每步 vanilla top-k 的各模型 logit）
      `output/dbg_imf_ours.jsonl.debug.jsonl`（可选：提供现有判据 D_step=全词表平均 var）
输出：`records/P1-3_offline_screen.csv` + 终端表格

⚠️ 使用边界（务必记住）
  1. 只在**这一条轨迹**上比较（新方案会改变后续轨迹；FSR/ACC 必须真跑）；
  2. τ 用**本 run 自身分布**的分位数（不是 clean 标定）→ 只用于**排序筛选**；
  3. k=200 的截断已实测 0% 影响（见 doc/P1-2诊断报告.md §9）。

指标：
  fp_top1_strong ↓ 融合后 argmax 落在"指纹侧且 max_delta≥θ"坐标上的步数（指纹驱动的输出，越少越好）
  flip_base      ↓ 翻盘且"被顶掉的坐标属于 base 侧"的步数（误伤）
  flip_fp        ↑ 翻盘且"被顶掉的坐标属于指纹侧"的步数（抑制生效）
  leak_gap       ↓ Σ(应有减幅 − 实际减幅)（"该罚未罚"量）
"""
import argparse
import csv
import json
from pathlib import Path

import torch

import gate_core

THETA_FP = 5.0      # "强指纹坐标"阈值（max_delta ≥ θ 视为指纹抬高）
SPIKE = 0.5         # "该模型在这个坐标上离群"的阈值（δ_i ≥ SPIKE 记一次 spike）
# 这些判据大量取 0（"非单一离群"的坐标）⇒ 分位阈值只在正值区间上取
POS_PCT_CRITERIA = {"solo", "tgt_solo"}

# 场景 → 指纹模型下标（决定"指纹侧/误伤"口径）
#   1fp：model1=指纹、model2/3=base×2 → 0
#   3fp：model1=IF(0) model2=Hash(1) model3=ImF(2)，靶子由所用测试集决定
FP_IDX_BY_STEM = {
    "dbg_imf_ours": 0, "dumpB_if": 0, "dumpB_hash": 0,
    "dumpA_if": 0, "dumpA_hash": 1, "dumpA_imf": 2,
}


def infer_fp_idx(path):
    """由文件名推断指纹模型下标；未匹配到时退回 0（= 与旧行为一致）。"""
    name = Path(path).name
    for stem, idx in FP_IDX_BY_STEM.items():
        if stem in name:
            return idx
    return 0


def load_jsonl(p):
    recs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def step_tensors(rec, fp_idx=0):
    """→ ids[k], van[k], max_delta[k], var[k], pen_full[k], fp_side[k], delta_tgt[k], delta[k,N]"""
    ids = [r[0] for r in rec["topn"]]
    L = torch.tensor([r[1] for r in rec["topn"]], dtype=torch.float32)      # [k,N]
    n = L.shape[1]
    van = L.mean(dim=1)
    others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
    delta = torch.clamp(L - others, min=0.0)
    max_delta = delta.max(dim=1).values
    var = L.var(dim=1, unbiased=True)                    # 与 ensemble_logit 内一致（无偏）
    pen_full = (L - van.unsqueeze(1)).abs().sum(dim=1) / (2 * (n - 1))     # α=1 时的应有减幅
    rest = L[:, [j for j in range(n) if j != fp_idx]].mean(dim=1)
    fp_side = (L[:, fp_idx] - rest) > 0.5                # 目标（指纹）模型在抬高该坐标
    return ids, van, max_delta, var, pen_full, fp_side, delta[:, fp_idx], delta


def solo_crit(delta, md, dt):
    """P1-3b 判据：只压"恰好一个模型离群"的坐标。

    K(v) = #{i: δ_i(v) ≥ SPIKE}；solo = max_i δ_i · 1[K=1]；tgt_solo = δ_target · 1[K=1]。
    判据定义与真跑代码共用 `gate_core.gate_criterion_values`（delta 形状 [k,N] → model_dim=1）。
    零点堆叠：K≠1 的坐标恒为 0（1fp-Hash 里占 98%），故分位阈值须在**正值**上取。
    """
    nsp = gate_core.solo_spike_count(delta, spike=SPIKE, model_dim=1)
    z = torch.zeros_like(md)
    solo = gate_core.gate_criterion_values(delta, criterion="solo", spike=SPIKE, model_dim=1)
    return solo, torch.where(nsp == 1, dt, z)




def quantile(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


CRITERIA = {           # 名字 -> (层级, 说明)
    # —— 逐坐标判据（在词表坐标上判断）——
    "max_delta":      ("coord", "max_i δ_i(v)：与抑制量同源的一阶量"),
    "var":            ("coord", "逐坐标跨模型方差（二阶）"),
    "pen_full":       ("coord", "该坐标的应有减幅 R(v)=Σ|d|/(2(N−1))（α=1）"),
    "md_over_margin": ("coord", "max_delta / (top1−top2 边际)：相对边际的强度（自校准）"),
    "combo_md_ad":    ("coord", "max_delta × 1[三模型 argmax 不一致]"),
    # —— 逐坐标判据（P1-3b 新增：单一模型离群）——
    "solo":           ("coord", "【P1-3b】solo=max_i δ_i·1[恰好一个模型 δ≥SPIKE]（分位在正值上取）"),
    "tgt_solo":       ("coord", "【P1-3b】solo ∩ 目标模型（δ_target·1[恰好一个模型离群]）"),
    # —— 步级判据（整步一个数）——
    "var_full":       ("step",  "【现有判据】全词表平均 var（D_step，来自 debug 文件）"),
    "var_topk":       ("step",  "top-k 内 var 的平均（原提案里的 var-topk）"),
    "max_var":        ("step",  "整步内最大 var"),
    "max_delta_step": ("step",  "整步内最大 max_delta（同判据、不同粒度，用于对照）"),
    "n_high_delta":   ("step",  "该步 max_delta ≥ θ(=5) 的坐标个数"),
    "argmax_diff":    ("step",  "三模型 argmax 是否不一致（0/1）"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topn", default=r"output\dbg_imf_ours.jsonl.topn.jsonl")
    ap.add_argument("--debug", default=r"output\dbg_imf_ours.jsonl.debug.jsonl")
    ap.add_argument("--out", default=r"records\P1-3_offline_screen.csv")
    ap.add_argument("--taus", default="85,90,95,99")
    ap.add_argument("--alphas", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--fp_idx", type=int, default=-1,
                    help="指纹模型下标（决定'指纹侧/误伤'口径）；-1=按文件名推断")
    args = ap.parse_args()

    steps = load_jsonl(args.topn)
    dstep = {}
    if Path(args.debug).exists():
        for r in load_jsonl(args.debug):
            dstep[(r["item"], r["step"])] = r["D_step"]
    print("载入 %d 个解码步；k=%d；debug 提供 D_step 的步数=%d"
          % (len(steps), steps[0]["k"], len(dstep)))

    cache = []
    fp_idx = args.fp_idx if args.fp_idx >= 0 else infer_fp_idx(args.topn)
    print("指纹模型下标 fp_idx=%d（%s）" % (fp_idx, "命令行指定" if args.fp_idx >= 0 else "按文件名推断"))
    for rec in steps:
        ids, van, md, var, pen, fps, dt, delta = step_tensors(rec, fp_idx)
        L = torch.tensor([r[1] for r in rec["topn"]], dtype=torch.float32)   # [k,N]
        am = [int(L[:, j].argmax()) for j in range(L.shape[1])]               # 各模型自己的 argmax
        t2 = torch.topk(van, 2).values
        margin = float(t2[0] - t2[1])
        dis_am = 1.0 if len(set(am)) > 1 else 0.0
        solo, tgt_solo = solo_crit(delta, md, dt)
        crit = {
            # 逐坐标
            "max_delta": md,
            "var": var,
            "pen_full": pen,
            "md_over_margin": md / (margin + 1e-6),
            "combo_md_ad": md * dis_am,
            "solo": solo,
            "tgt_solo": tgt_solo,
            # 步级
            "max_var": torch.tensor([float(var.max())]),
            "var_topk": torch.tensor([float(var.mean())]),
            "max_delta_step": torch.tensor([float(md.max())]),
            "n_high_delta": torch.tensor([float((md >= THETA_FP).sum())]),
            "argmax_diff": torch.tensor([dis_am]),
        }
        cache.append(dict(ids=ids, van=van, md=md, var=var, pen=pen, fps=fps, dt=dt,
                          crit=crit, D_step=dstep.get((rec["item"], rec["step"]))))
    print("判据预算完成。")

    taus = [float(x) for x in args.taus.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    rows = []

    def run_variant(name, level, crit_name, tau, alpha):
        fp_strong = flip_base = flip_fp = gated = 0
        leak = 0.0
        for c in cache:
            van, pen, fps, dt = c["van"], c["pen"], c["fps"], c["dt"]
            if crit_name == "none":
                mask = torch.ones_like(van, dtype=torch.bool)
            elif level == "coord":
                mask = c["crit"][crit_name] > tau
            else:
                v = c["D_step"] if crit_name == "var_full" else float(c["crit"][crit_name].item())
                mask = torch.full_like(van, bool(v is not None and v > tau), dtype=torch.bool)
            if bool(mask.any()):
                gated += 1
            penalty = alpha * pen * mask.float()
            fused = van - penalty
            leak += float((pen - penalty).clamp(min=0).sum())
            a_van, a_fus = int(van.argmax()), int(fused.argmax())
            if bool(fps[a_fus]) and float(dt[a_fus]) >= THETA_FP:
                fp_strong += 1
            if a_van != a_fus:
                if bool(fps[a_van]):
                    flip_fp += 1
                else:
                    flip_base += 1
        rows.append(dict(variant=name, level=level, crit=crit_name, tau=round(tau, 4), alpha=alpha,
                         gated_step_ratio=round(gated / len(cache), 4),
                         fp_top1_strong=fp_strong, flip_base=flip_base, flip_fp=flip_fp,
                         leak_gap=round(leak, 1)))

    run_variant("vanilla（无抑制）", "none", "none", float("nan"), 0.0)
    for a in (0.5, 1.0, 2.0):        # 无门控的 ours（现在就用的方法）= 关键参考基线
        run_variant("ours（无门控，逐坐标）_a%g" % a, "none", "none", float("nan"), a)
    for crit_name, (level, _desc) in CRITERIA.items():
        vals = []
        for c in cache:
            if level == "coord":
                vals += [float(x) for x in c["crit"][crit_name]]
            else:
                v = c["D_step"] if crit_name == "var_full" else float(c["crit"][crit_name].item())
                if v is not None:
                    vals.append(float(v))
        # solo / tgt_solo 有零点堆叠（K≠1 的坐标恒为 0，1fp-Hash 里占 98%）
        #   → 分位阈值必须在**正值**上取，否则 p90 会取到 0 而门全开。
        if crit_name in POS_PCT_CRITERIA:
            vals = [v for v in vals if v > 0.0]
        if not vals:
            continue
        for tp in taus:
            tau = quantile(vals, tp / 100.0)
            for a in alphas:
                run_variant("%s_%s_p%d%s_a%g" %
                            (crit_name, level, tp,
                             "_pv" if crit_name in POS_PCT_CRITERIA else "", a),
                            level, crit_name, tau, a)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n=== 变体结果（按 fp_top1强↓、误伤flip↓、抑制flip↑ 排序）→ %s ===" % out)
    print("%-30s %8s %8s %12s %10s %10s %10s" %
          ("variant", "门开步比", "τ", "fp_top1强↓", "误伤flip↓", "抑制flip↑", "leak_gap↓"))
    for r in sorted(rows, key=lambda x: (x["fp_top1_strong"], x["flip_base"], -x["flip_fp"]))[:25]:
        print("%-30s %8.3f %8.3f %12d %10d %10d %10.0f" %
              (r["variant"], r["gated_step_ratio"], r["tau"], r["fp_top1_strong"],
               r["flip_base"], r["flip_fp"], r["leak_gap"]))
    print("\n=== 每个判据的最佳表现（fp_top1强 最小、再看误伤）===")
    print("%-16s %-7s %-32s %9s %7s %7s %9s" %
          ("判据", "粒度", "最佳变体", "fp_top1强↓", "误伤↓", "抑制↑", "leak_gap↓"))
    for cname, (lvl, _desc) in CRITERIA.items():
        sub = [r for r in rows if r["crit"] == cname]
        if not sub:
            continue
        b = min(sub, key=lambda x: (x["fp_top1_strong"], x["flip_base"]))
        print("%-16s %-7s %-32s %9d %7d %7d %9.0f" %
              (cname, lvl, b["variant"], b["fp_top1_strong"], b["flip_base"],
               b["flip_fp"], b["leak_gap"]))

    print("\n=== 参考基线 ===")
    print("%-30s %8s %8s %12s %10s %10s %10s" %
          ("variant", "门开步比", "τ", "fp_top1强↓", "误伤flip↓", "抑制flip↑", "leak_gap↓"))
    for r in rows:
        if r["crit"] == "none":
            print("%-30s %8.3f %8s %12d %10d %10d %10.0f" %
                  (r["variant"], r["gated_step_ratio"], "-", r["fp_top1_strong"],
                   r["flip_base"], r["flip_fp"], r["leak_gap"]))

    print("\n=== 现有判据(var_full, 步级) vs 候选(max_delta, 逐坐标)：同 α 对照 ===")
    print("%-34s %10s %12s %10s" % ("variant", "fp_top1强↓", "误伤flip↓", "leak_gap↓"))
    for r in rows:
        if r["crit"] in ("var_full", "max_delta") and r["alpha"] in (1.0, 2.0) and r["tau"] > 0:
            print("%-34s %10d %12d %10.0f" %
                  (r["variant"], r["fp_top1_strong"], r["flip_base"], r["leak_gap"]))

    print("\n=== Pareto 前沿（在给定误伤上限内，找 fp_top1强 最小者）===")
    gated = [r for r in rows if r["crit"] != "none"]
    for cap in (0, 1, 2, 5, 9, 16):
        cand = [r for r in gated if r["flip_base"] <= cap]
        if not cand:
            print("  误伤 ≤ %-3d ：无可行变体" % cap)
            continue
        best = min(cand, key=lambda x: x["fp_top1_strong"])
        print("  误伤 ≤ %-3d ：最佳 %-30s fp_top1强=%3d  抑制flip=%3d  leak_gap=%.0f"
              % (cap, best["variant"], best["fp_top1_strong"], best["flip_fp"], best["leak_gap"]))



if __name__ == "__main__":
    main()

