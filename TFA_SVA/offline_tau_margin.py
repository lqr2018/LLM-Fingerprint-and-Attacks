# -*- coding: utf-8 -*-
"""τ 的残差代价量化：α=1、τ>0 时"超出部分被削、残差 τ 保留"对**指纹坐标边际**的影响。

理论预言（本脚本要验证）：
  只有 argmax 那一个模型被削 p = α(x₍₁₎ − x₍₂₎ − τ) ⇒ 指纹坐标的融合值相对 τ=0 的情形
  **恰好回升 τ/N**（N=3 ⇒ τ/3）。若该坐标原本的领先边际（fused[fp] − 最佳非指纹坐标）
  ≤ τ/N，它就会重新被竞争对手超过。

指标（逐步；只统计"vanilla 的 argmax 是强指纹坐标"的步）：
  fp_win      = 抑制后指纹坐标仍然胜出的步数（= 旧的 fp_strong）
  边际中位数   = fused[fp] − max_{非指纹侧} fused
  回升量      = (τ=某值时边际) − (τ=0 时边际)，应 ≈ τ/N
用法：python TFA_SVA/offline_tau_margin.py → output/tau_margin.txt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_criterion_recheck import SCENES, THETA_FP, load_jsonl  # noqa: E402
from offline_adaptive_screen import op_rawgap  # noqa: E402


def build(stem, fp_idx):
    cache = []
    for rec in load_jsonl(stem + ".topn.jsonl"):
        L = torch.tensor([r[1] for r in rec["topn"]], dtype=torch.float32)
        n = L.shape[1]
        others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
        d = torch.clamp(L - others, min=0.0)
        rest = L[:, [j for j in range(n) if j != fp_idx]].mean(dim=1)
        van = L.mean(dim=1)
        a_van = int(van.argmax())
        cache.append(dict(L=L, van=van, d_t=d[:, fp_idx],
                          side_all=(L[:, fp_idx] - rest) > 0.5,
                          is_fp_step=bool((L[:, fp_idx] - rest)[a_van] > 0.5)
                          and float(d[a_van, fp_idx]) >= THETA_FP))
    return cache


def stats(cache, tau, alpha=1.0):
    win = tot = 0
    margins = []
    for c in cache:
        if not c["is_fp_step"]:
            continue
        tot += 1
        fused = op_rawgap(c["L"], tau=tau, alpha=alpha)
        a_van = int(c["van"].argmax())
        others = fused.clone()
        others[a_van] = float("-inf")
        margin = float(fused[a_van] - others.max())
        margins.append(margin)
        if int(fused.argmax()) == a_van:
            win += 1
    m = torch.tensor(margins) if margins else torch.zeros(1)
    return win, tot, float(m.median()), float(m.mean())


def fp_value_shift(cache, tau, alpha=1.0, n_models=3):
    """在"指纹坐标"上，融合值相对 τ=0 的回升量（逐点精确检验 τ/N 主项）。"""
    shifts = []
    for c in cache:
        if not c["is_fp_step"]:
            continue
        a_van = int(c["van"].argmax())
        f0 = float(op_rawgap(c["L"], tau=0.0, alpha=alpha)[a_van])
        ft = float(op_rawgap(c["L"], tau=tau, alpha=alpha)[a_van])
        shifts.append(ft - f0)
    s = torch.tensor(shifts) if shifts else torch.zeros(1)
    return float(s.mean()), float(s.median())


def main():
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    TAUS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    emit("τ 的残差代价（α=1，间隙版）：只统计 vanilla argmax 为强指纹坐标的步")
    emit("=" * 104)
    for name, (stem, fidx) in SCENES.items():
        cache = build(stem, fidx)
        n_models = cache[0]["L"].shape[1]
        base_m = stats(cache, 0.0)[2]
        emit("")
        emit("[%s] 步=%d N=%d 指纹驱动步=%d（τ=0 边际中位数=%.4f）"
             % (name, len(cache), n_models, stats(cache, 0.0)[1], base_m))
        emit("  %-8s %-8s %-10s %-12s %-12s %-10s" % ("τ", "仍胜出步", "总步", "边际中位数", "边际均值", "回升(中位)"))
        for t in TAUS:
            win, tot, med, mean = stats(cache, t)
            emit("  %-8.2f %-8d %-10d %-12.4f %-12.4f %-10s (理论 τ/N=%.4f)"
                 % (t, win, tot, med, mean,
                    "—" if t == 0 else "%+.4f" % (med - base_m), t / n_models))
        mshift, medshift = fp_value_shift(cache, 3.0)
        emit("  校验：τ=3.0 时指纹坐标**自身融合值**回升 均值=%.4f / 中位=%.4f（理论主项 τ/N=%.4f）"
             % (mshift, medshift, 3.0 / n_models))
        emit("  α=2 对照：")
        for t in (0.0, 1.0, 3.0):
            win, tot, med, mean = stats(cache, t, alpha=2.0)
            emit("    τ=%.2f α=2 → 仍胜出 %d/%d，边际中位 %.4f（较 α=1 的 %.4f）"
                 % (t, win, tot, med, stats(cache, t, alpha=1.0)[2]))
        emit("-" * 104)

    out = Path(r"output\tau_margin.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n已写出：%s" % out)


if __name__ == "__main__":
    main()
