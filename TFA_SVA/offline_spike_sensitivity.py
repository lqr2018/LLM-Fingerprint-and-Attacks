# -*- coding: utf-8 -*-
"""spike 超参敏感性：`solo` 判据里的"离群阈值" spike 到底影响什么（6 条冻结轨迹，纯 CPU）。

动机：`spike` 最初是从诊断代码里沿用的常量（0.5），**没有标定程序**。本脚本回答两件事：
  1. 在 1fp（两个 base 副本相同）下，spike 是否**根本不起作用**（数学猜想：等价于给 τ 加下限）；
  2. 在 3fp 下，spike 如何改变"单一离群"集合，以及门控/误伤/抑制指标对它的敏感度；
  3. 与"spike ≡ τ"（把两个阈值合并成一个，少一个超参）的对比。

用法（仓库根目录）：python TFA_SVA/offline_spike_sensitivity.py
输出：output/spike_sensitivity.txt
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_core as G  # noqa: E402
from offline_criterion_recheck import SCENES, THETA_FP, load_jsonl  # noqa: E402


def build(stem, fp_idx):
    cache = []
    for rec in load_jsonl(stem + ".topn.jsonl"):
        L = torch.tensor([r[1] for r in rec["topn"]], dtype=torch.float32)   # [k,N]
        n = L.shape[1]
        van = L.mean(dim=1)
        others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
        d = torch.clamp(L - others, min=0.0)                                 # [k,N]
        rest = L[:, [j for j in range(n) if j != fp_idx]].mean(dim=1)
        cache.append(dict(van=van, d_t=d[:, fp_idx], pen=(L - van.unsqueeze(1)).abs().sum(dim=1) / (2 * (n - 1)),
                          tgt_side=(d[:, fp_idx] >= 0.5), d=d))
    return cache


def run(cache, spikes, taus, s_is_tau=False):
    """返回 {(spike, τ): (门控数, 门控中目标侧数, 指纹驱动步, 真误伤)}"""
    out = {}
    for s in spikes:
        for t in taus:
            g = g_tgt = fp_strong = harm = flip = 0
            for c in cache:
                d, n = c["d"], c["van"].numel()
                s_eff = t if s_is_tau else s
                crit = G.gate_criterion_values(d, criterion="solo", spike=s_eff, model_dim=1)
                mask = crit > t
                g += int(mask.sum())
                g_tgt += int((mask & c["tgt_side"]).sum())
                fused = c["van"] - 1.0 * c["pen"] * mask.float()
                a_van, a_fus = int(c["van"].argmax()), int(fused.argmax())
                if bool(c["tgt_side"][a_fus]) and float(c["d_t"][a_fus]) >= THETA_FP:
                    fp_strong += 1
                if a_van != a_fus:
                    flip += 1
                    # 真误伤：被顶掉的是"多数共识"坐标（≥2 个模型 δ ≥ 0.5）
                    if int((d[a_van] >= 0.5).sum()) >= 2:
                        harm += 1
            out[(s, t)] = (g, g_tgt, fp_strong, harm, flip)
    return out


SPIKES = [0.05, 0.25, 0.5, 1.0, 2.0, 5.0]
TAUS = [1.0, 2.0, 3.55, 5.0]


def main():
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("spike 敏感性（solo 判据；格 = 门控坐标 / 门控中目标侧 / 指纹驱动步 / 真误伤）")
    emit("=" * 108)
    for name, (stem, fidx) in SCENES.items():
        cache = build(stem, fidx)
        res = run(cache, SPIKES, TAUS)
        emit("")
        emit("[%s] 步=%d 靶子=下标%d" % (name, len(cache), fidx))
        hdr = "%-8s" % "spike\\τ"
        for t in TAUS:
            hdr += " %24s" % ("τ=%.2f" % t)
        emit(hdr)
        for s in SPIKES:
            row = "%-8.2f" % s
            for t in TAUS:
                g, gt, fs, hm, fl = res[(s, t)]
                row += " %24s" % ("%d/%d/%d/%d" % (g, gt, fs, hm))
            emit(row)
        # 1fp：spike 不变性检验（同一 τ 下，不同 spike 的门控集合是否完全相同）
        inv = True
        for t in TAUS:
            base = res[(SPIKES[0], t)][0]
            for s in SPIKES[1:]:
                if res[(s, t)][0] != base:
                    inv = False
        emit("  ⇒ 同一 τ 下 spike 改变门控数：%s" % ("否（spike 不起作用）" if inv else "是（spike 有影响）"))
        # spike ≡ τ（合并成一个超参）
        res2 = run(cache, TAUS, TAUS, s_is_tau=True)
        row = "%-8s" % "s≡τ"
        for t in TAUS:
            g, gt, fs, hm, fl = res2[(t, t)]
            row += " %24s" % ("%d/%d/%d/%d" % (g, gt, fs, hm))
        emit(row)
        emit("-" * 108)

    out = Path(r"output\spike_sensitivity.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n已写出：%s" % out)


if __name__ == "__main__":
    main()
