# -*- coding: utf-8 -*-
"""P1-3 离线筛选（跨场景批量版）：对多份 `<*.topn.jsonl>` 各跑一遍同一网格，输出对照。

用法：
    python TFA_SVA/offline_gate_screen_batch.py --glob "output/dump*.topn.jsonl"
    # 单场景可直接用 offline_gate_screen.py

输出：`records/P1-3_offline_screen_all.csv`（含 scene 列）+ 终端"每场景 × 每判据最佳"对照表。
复用 `offline_gate_screen.py` 的判据定义与张量计算，保证两个脚本口径一致。
"""
import argparse
import csv
import glob
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_gate_screen import (CRITERIA, POS_PCT_CRITERIA, THETA_FP,  # noqa: E402
                                 infer_fp_idx, load_jsonl, quantile, solo_crit,
                                 step_tensors)


def build_cache(steps, dstep, fp_idx):
    cache = []
    for rec in steps:
        ids, van, md, var, pen, fps, dt, delta = step_tensors(rec, fp_idx)
        L = torch.tensor([r[1] for r in rec["topn"]], dtype=torch.float32)
        am = [int(L[:, j].argmax()) for j in range(L.shape[1])]
        t2 = torch.topk(van, 2).values
        margin = float(t2[0] - t2[1])
        dis_am = 1.0 if len(set(am)) > 1 else 0.0
        solo, tgt_solo = solo_crit(delta, md, dt)
        crit = {
            "max_delta": md, "var": var, "pen_full": pen,
            "md_over_margin": md / (margin + 1e-6), "combo_md_ad": md * dis_am,
            "solo": solo, "tgt_solo": tgt_solo,
            "max_var": torch.tensor([float(var.max())]),
            "var_topk": torch.tensor([float(var.mean())]),
            "max_delta_step": torch.tensor([float(md.max())]),
            "n_high_delta": torch.tensor([float((md >= THETA_FP).sum())]),
            "argmax_diff": torch.tensor([dis_am]),
        }
        cache.append(dict(ids=ids, van=van, md=md, var=var, pen=pen, fps=fps, dt=dt,
                          crit=crit, D_step=dstep.get((rec["item"], rec["step"]))))
    return cache


def run_variant(cache, name, level, crit_name, tau, alpha):
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
    return dict(variant=name, level=level, crit=crit_name, tau=round(tau, 4), alpha=alpha,
                gated_step_ratio=round(gated / len(cache), 4), n_steps=len(cache),
                fp_top1_strong=fp_strong, flip_base=flip_base, flip_fp=flip_fp,
                leak_gap=round(leak, 1))


def screen_one(topn_path, debug_path, taus, alphas, fp_idx=None):
    steps = load_jsonl(topn_path)
    dstep = {}
    if Path(debug_path).exists():
        for r in load_jsonl(debug_path):
            dstep[(r["item"], r["step"])] = r["D_step"]
    if fp_idx is None:
        fp_idx = infer_fp_idx(topn_path)
    cache = build_cache(steps, dstep, fp_idx)
    rows = [run_variant(cache, "vanilla", "none", "none", float("nan"), 0.0),
            run_variant(cache, "ours_a1", "none", "none", float("nan"), 1.0)]
    for crit_name, (level, _d) in CRITERIA.items():
        vals = []
        for c in cache:
            if level == "coord":
                vals += [float(x) for x in c["crit"][crit_name]]
            else:
                v = c["D_step"] if crit_name == "var_full" else float(c["crit"][crit_name].item())
                if v is not None:
                    vals.append(float(v))
        if not vals:
            continue
        if crit_name in POS_PCT_CRITERIA:      # solo 类判据：分位只在正值上取（零点堆叠）
            vals = [v for v in vals if v > 0.0]
            if not vals:
                continue
        for tp in taus:
            tau = quantile(vals, tp / 100.0)
            for a in alphas:
                rows.append(run_variant(cache, "%s_%s_p%d%s_a%g" %
                                        (crit_name, level, tp,
                                         "_pv" if crit_name in POS_PCT_CRITERIA else "", a),
                                        level, crit_name, tau, a))
    return rows, len(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=r"output\dump*.topn.jsonl",
                    help="匹配多份 <*.topn.jsonl>（也支持单个文件）")
    ap.add_argument("--out", default=r"records\P1-3_offline_screen_all.csv")
    ap.add_argument("--taus", default="85,90,95,99")
    ap.add_argument("--alphas", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--top", type=int, default=12, help="每场景打印前 N 行（按 fp_top1强、误伤排序）")
    ap.add_argument("--fp_idx", type=int, default=-1,
                    help="强制指定指纹模型下标（覆盖按文件名推断）")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print("未匹配到文件：%s" % args.glob)
        return
    taus = [float(x) for x in args.taus.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]

    all_rows = []
    for f in files:
        scene = Path(f).name.replace(".topn.jsonl", "").replace(".jsonl", "")
        dbg = f.replace(".topn.jsonl", ".debug.jsonl")
        rows, n = screen_one(f, dbg, taus, alphas, args.fp_idx if args.fp_idx >= 0 else None)
        for r in rows:
            r["scene"] = scene
            r["fp_idx"] = args.fp_idx if args.fp_idx >= 0 else infer_fp_idx(f)
        all_rows += rows
        print("[%s] 步数=%d，变体=%d（debug: %s，fp_idx=%d）"
              % (scene, n, len(rows), "有" if Path(dbg).exists() else "无",
                 args.fp_idx if args.fp_idx >= 0 else infer_fp_idx(f)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["scene"] + [k for k in all_rows[0].keys() if k != "scene"]
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)

    print("\n=== 每场景 × 每判据 的最佳表现（fp_top1强 最小、再看误伤）→ %s ===" % out)
    print("%-16s %-16s %-7s %-30s %8s %7s %7s" %
          ("scene", "判据", "粒度", "最佳变体", "fp强↓", "误伤↓", "抑制↑"))
    for scene in sorted({r["scene"] for r in all_rows}):
        for cname, (lvl, _d) in CRITERIA.items():
            sub = [r for r in all_rows if r["scene"] == scene and r["crit"] == cname]
            if not sub:
                continue
            b = min(sub, key=lambda x: (x["fp_top1_strong"], x["flip_base"]))
            print("%-16s %-16s %-7s %-30s %8d %7d %7d" %
                  (scene, cname, lvl, b["variant"], b["fp_top1_strong"], b["flip_base"], b["flip_fp"]))
        print("-" * 110)

    print("\n=== 跨场景一致性检查：三档（max_delta 逐坐标）在各场景的排序 ===")
    print("%-16s %8s %8s %8s %10s %10s" %
          ("scene", "档1(P90a1)", "档2(P95a2)", "档3(P90a2)", "vanilla", "ours_a1"))
    for scene in sorted({r["scene"] for r in all_rows}):
        def get(name):
            r = next((x for x in all_rows if x["scene"] == scene and x["variant"] == name), None)
            return r["fp_top1_strong"] if r else -1
        print("%-16s %8d %8d %8d %10d %10d" %
              (scene, get("max_delta_coord_p90_a1"), get("max_delta_coord_p95_a2"),
               get("max_delta_coord_p90_a2"), get("vanilla"), get("ours_a1")))


if __name__ == "__main__":
    main()

