# -*- coding: utf-8 -*-
"""超额抑制（runner-up gating）在**任意模型数 N** 下的性质检查（纯 CPU，随机 + 构造用例）。

待验证命题：
  Q1 只有 δ 最大的那 1 个模型被削弱（其余 p_j = 0）—— 对任意 N 成立？
  Q2 "其余 N−1 个模型相等时，α=1、τ=0 恰好把最高模型压回它们的公共值" —— 对任意 N 成立？
  Q3 削弱量随"顶部间隙"单调增（第二名的 δ 越大 → 削弱越弱）—— 一般 N 成立？
  Q4 δ 的尺度因子 = N/(N−1)，以及两种态：第二名在均值上方 / 下方。
  Q5 两个模型"撞车"（并列最高）时完全不动作（共识零抑制）。

用法：python TFA_SVA/check_excess_ngeneral.py → output/excess_ngeneral.txt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def excess_suppress(L, alpha=1.0, tau=0.0):
    """L=[k,N] → (融合值[k], 抑制向量 p[k,N], δ[k,N])"""
    n = L.shape[1]
    others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
    d = torch.clamp(L - others, min=0.0)
    d2 = d.sort(dim=1, descending=True).values[:, 1] if n >= 2 else torch.zeros(L.shape[0])
    p = alpha * torch.clamp(d - (d2 + tau).unsqueeze(1), min=0.0)
    return (L - p).mean(dim=1), p, d


def main():
    out = []

    def emit(s=""):
        out.append(s)
        print(s)

    torch.manual_seed(0)
    emit("Q1 只有 δ 最大的模型被削弱（随机 2000 个向量 × N=2..8）")
    for n in range(2, 9):
        L = torch.randn(2000, n) * 3
        _, p, d = excess_suppress(L, alpha=1.0, tau=0.0)
        argmax_d = d.argmax(dim=1)
        others = p.clone()
        others[torch.arange(L.shape[0]), argmax_d] = 0.0
        emit("  N=%d：其余模型的 p 最大绝对值 = %.3e %s"
             % (n, float(others.abs().max()), "✓" if float(others.abs().max()) < 1e-9 else "✗"))

    emit("")
    emit("Q2 其余 N−1 个模型相等时，α=1、τ=0 把最高模型压回它们的公共值 m")
    for n in range(2, 9):
        m = 1.0
        for gap in (3.0, 9.0):
            L = torch.full((1, n), m, dtype=torch.float32)
            L[0, 0] = m + gap                                   # 第 0 个模型独自高
            fused, p, d = excess_suppress(L, alpha=1.0, tau=0.0)
            emit("  N=%d 间隙=%4.1f：被削后该模型 = %.6f（期望 m=%.1f），p=%.4f"
                 % (n, gap, float(L[0, 0] - p[0, 0]), m, float(p[0, 0])))

    emit("")
    emit("Q3 削弱量随顶部间隙单调增（N=4 与 N=3 对比）")
    for n in (3, 4, 5):
        row = []
        for gap in (0.0, 1.0, 2.0, 5.0):
            L = torch.zeros(1, n)
            L[0, 0] = 10.0
            L[0, 1] = 10.0 - gap                                # 第二名与第一名差 gap
            L[0, 2:] = 0.0
            _, p, _ = excess_suppress(L, alpha=1.0, tau=0.0)
            row.append("间隙%.0f→p=%.3f" % (gap, float(p[0, 0])))
        emit("  N=%d：%s" % (n, "  ".join(row)))

    emit("")
    emit("Q4 δ 的尺度因子 = N/(N−1)（用 [10, 1, 1, ...] 检验 δ₁₎）")
    for n in range(2, 9):
        L = torch.full((1, n), 1.0)
        L[0, 0] = 10.0
        _, _, d = excess_suppress(L)
        mean = float(L.mean())
        emit("  N=%d：δ₍₁₎=%.4f，N/(N−1)·(10−均值)=%.4f，比值=%.4f"
             % (n, float(d[0, 0]), n / (n - 1) * (10.0 - mean), float(d[0, 0]) / (n / (n - 1) * (10.0 - mean))))

    emit("")
    emit("Q5 两个模型并列最高（撞车）= 共识 ⇒ 零抑制")
    for n in (3, 4, 5):
        L = torch.full((1, n), 1.0)
        L[0, 0] = L[0, 1] = 10.0
        fused, p, _ = excess_suppress(L, alpha=2.0, tau=0.0)
        emit("  N=%d：[10,10,1,...] → 总抑制量 Σp = %.3e（应为 0）" % (n, float(p.sum())))

    out_p = Path(r"output\excess_ngeneral.txt")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("\n".join(out), encoding="utf-8")
    print("\n已写出：%s" % out_p)


if __name__ == "__main__":
    main()
