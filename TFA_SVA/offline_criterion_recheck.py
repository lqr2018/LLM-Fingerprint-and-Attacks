# -*- coding: utf-8 -*-
"""P1-3b 判据重审：在 6 条已冻结轨迹上重比"坐标级门控判据"（纯 CPU，只读诊断）。

背景（为什么要重审）
  1. 旧筛选（offline_gate_screen*.py）把指纹模型**硬编码为下标 0**；3fp 场景里
     Hash=下标1、ImF=下标2 ⇒ 那两行的"指纹侧/误伤"口径错误。本脚本按场景指定 fp_idx。
  2. 旧"误伤"= 被顶掉的坐标不在"目标指纹侧"。但 3fp 里另外两个模型各自也带指纹，
     "非目标侧"≠"干净共识侧"。本脚本新增**多数共识误伤**：被顶掉的坐标若 ≥2/3 个模型
     同时抬高（δ≥SPIKE），才计为真误伤。
  3. 旧比较用"同一 τ"，但各判据量纲不同（logit / 平方 / 计数 / 无量纲），不可比。
     改用**绝对 τ 网格 + 同场景同 τ**，并同时报告门控规模，便于看"同抑制下的代价"。

候选判据
  loo_max   max_i ReLU(l_i − l̄_{−i})           ← 当前实现 maxdelta_gate
  solo      仅当"恰好一个模型 δ≥SPIKE"时取 max_i δ_i，否则 0
  tgt       δ_target = ReLU(l_target − 其余模型均值)（1fp 下等价于中位数参照）
  tgt_solo  tgt ∩ "恰好一个模型 δ≥SPIKE"
  var_full  整步门控（现有 D_step，来自 debug 文件）
说明：solo = ∪_i tgt_solo_i（一个坐标最多只可能是某一个模型的"单点离群"），
      因此 solo 不需要知道"哪个模型是指纹模型"，是可部署形式。

输出：`output/criterion_recheck.txt` + `records/P1-3b_criterion_recheck.csv`
"""
import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_core  # noqa: E402
from offline_gate_screen import THETA_FP  # noqa: E402

# 场景 → (topn/jsonl 文件主干, 指纹模型下标)
#   1fp：model1=指纹, model2/3=base×2 → 0
#   3fp：model1=IF(0) model2=Hash(1) model3=ImF(2)，按测试集确定靶子
SCENES = OrderedDict([
    ("1fp-ImF",  ("output/dbg_imf_ours.jsonl", 0)),
    ("1fp-Hash", ("output/dumpB_hash.jsonl", 0)),
    ("1fp-IF",   ("output/dumpB_if.jsonl", 0)),
    ("3fp-IF",   ("output/dumpA_if.jsonl", 0)),
    ("3fp-Hash", ("output/dumpA_hash.jsonl", 1)),
    ("3fp-ImF",  ("output/dumpA_imf.jsonl", 2)),
])
TAUS = [0.5, 1.0, 2.0, 3.55, 5.0, 8.0]
ALPHA = 1.0
SPIKE = 0.5          # δ ≥ SPIKE 视为"该模型在这个坐标上离群"
CRITERIA = ["loo_max", "solo", "tgt", "tgt_solo", "var_full"]


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]



def build_cache(stem, fp_idx):
    topn_p = Path(stem + ".topn.jsonl")
    dbg_p = Path(stem + ".debug.jsonl")
    dstep = {}
    if dbg_p.exists():
        for r in load_jsonl(dbg_p):
            dstep[(r["item"], r["step"])] = r["D_step"]
    cache = []
    for rec in load_jsonl(topn_p):
        L = torch.tensor([r[1] for r in rec["topn"]], dtype=torch.float32)     # [k,N]
        n = L.shape[1]
        van = L.mean(dim=1)
        others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
        d = torch.clamp(L - others, min=0.0)
        d_t = d[:, fp_idx]
        n_spike = gate_core.solo_spike_count(d, spike=SPIKE, model_dim=1)
        md = d.max(dim=1).values
        z = torch.zeros_like(md)
        cache.append(dict(
            van=van, d_t=d_t, pen=(L - van.unsqueeze(1)).abs().sum(dim=1) / (2 * (n - 1)),
            tgt_side=(d_t >= SPIKE), majority=(n_spike >= 2),
            D=dstep.get((rec["item"], rec["step"])),
            crit={"loo_max": md, "solo": gate_core.gate_criterion_values(
                      d, criterion="solo", spike=SPIKE, model_dim=1),
                  "tgt": d_t, "tgt_solo": torch.where(n_spike == 1, d_t, z)}))
    return cache


def evaluate(cache, key, tau):
    g = g_tgt = g_maj = 0
    fp_strong = flip = flip_other = harm = 0
    for c in cache:
        n = int(c["van"].numel())
        if key == "var_full":
            on = c["D"] is not None and float(c["D"]) > tau
            mask = torch.full((n,), bool(on), dtype=torch.bool)
        else:
            mask = c["crit"][key] >= tau
        g += int(mask.sum())
        g_tgt += int((mask & c["tgt_side"]).sum())
        g_maj += int((mask & c["majority"]).sum())
        fused = c["van"] - ALPHA * c["pen"] * mask.float()
        a_van, a_fus = int(c["van"].argmax()), int(fused.argmax())
        if bool(c["tgt_side"][a_fus]) and float(c["d_t"][a_fus]) >= THETA_FP:
            fp_strong += 1
        if a_van != a_fus:
            flip += 1
            flip_other += int(not bool(c["tgt_side"][a_van]))
            harm += int(bool(c["majority"][a_van]))
    return dict(gated=g, gated_tgt=g_tgt, gated_maj=g_maj, fp_strong=fp_strong,
                flip=flip, flip_other=flip_other, harm=harm)


def main():
    global ALPHA
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt", default=r"output\criterion_recheck.txt")
    ap.add_argument("--csv", default=r"records\P1-3b_criterion_recheck.csv")
    ap.add_argument("--taus", default=",".join(str(t) for t in TAUS))
    ap.add_argument("--alpha", type=float, default=ALPHA, help="抑制强度 α（默认 1.0）")
    args = ap.parse_args()
    taus = [float(x) for x in args.taus.split(",")]
    ALPHA = args.alpha

    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("P1-3b 判据重审（6 条冻结轨迹，纯 CPU，只读）  α=%.2f" % ALPHA)
    emit("fp_strong↓ 融合后 argmax 落在\"目标模型抬高 δ_target≥%.0f\"的步数；" % THETA_FP)
    emit("harm↓ 被顶掉的是**多数共识**坐标（≥2/3 模型 δ≥%.1f）= 真误伤；" % SPIKE)
    emit("误伤other = 被顶掉的坐标不在目标侧（旧口径；3fp 下含别的模型的指纹，不等于有害）")
    emit("solo = ∪_i tgt_solo_i（一个坐标最多只可能是某一个模型的单点离群）→ 无需知道靶子是谁")
    rows = []
    for scene, (stem, fp_idx) in SCENES.items():
        cache = build_cache(stem, fp_idx)
        n_co = sum(int(c["van"].numel()) for c in cache)
        b0 = evaluate(cache, "solo", 1e9)
        emit("")
        emit("=" * 118)
        emit("[%s] 步=%d 坐标=%d 靶子=下标%d | 基线: 目标指纹驱动步=%d" %
             (scene, len(cache), n_co, fp_idx, b0["fp_strong"]))
        emit("%-9s %6s %9s %8s %8s %10s %7s %10s %6s" %
             ("判据", "τ", "门控", "门控tgt", "门控maj", "指纹驱动步", "翻盘", "误伤other", "真误伤"))
        for key in CRITERIA:
            for tau in taus:
                m = evaluate(cache, key, tau)
                emit("%-9s %6.2f %9d %8d %8d %10d %7d %10d %6d" %
                     (key, tau, m["gated"], m["gated_tgt"], m["gated_maj"], m["fp_strong"],
                      m["flip"], m["flip_other"], m["harm"]))
                rows.append(dict(scene=scene, n_steps=len(cache), n_coords=n_co, fp_idx=fp_idx,
                                 baseline_fp_strong=b0["fp_strong"], crit=key, tau=tau, **m))
        emit("-" * 118)

    txt = Path(args.txt)
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text("\n".join(lines), encoding="utf-8")
    csv_p = Path(args.csv)
    csv_p.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_p, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n已写出：%s\n        %s" % (txt, csv_p))


if __name__ == "__main__":
    main()
