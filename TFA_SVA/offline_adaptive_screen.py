# -*- coding: utf-8 -*-
"""自适应抑制（runner-up gating）的离线验证：p_i = α·ReLU(d_i − d₍₂₎ − τ)。

动机（来自 §9.2 的诊断）：原算子 `p_i = α·ReLU(d_i)` 把"两个模型一起高"的坐标也压了
（1fp 的 base 侧 `d=[0,Δ/2,Δ/2]`）⇒ Hash 退化。最小修正 = **只惩罚超过"第二名偏离"的那部分**。

与中位数方案的区别：不用中位数（N=3 时中位数会退化成均值/取中间模型，且 N 消融难做），
而是用"第二大的正偏离 d₍₂₎"作为**同为异常**的参照 ⇒ 定义良好、随 N 自然推广、置换不变。

性质（见 doc §12）：
  P1 共识零抑制：≥2 个模型并列最高 ⇒ d_i − d₍₂₎ = 0 ⇒ p = 0；
  P2 τ→0,α=1 且忽略 d₍₂₎ 时退化为原方法 `ours`；
  P3 置换不变（靶子无关）；P4 p 关于 d_i 单调增、关于 d₍₂₎ 单调减（自适应）；
  P5 自动只作用于唯一离群者（其余模型 d_j ≤ d₍₂₎ ⇒ p_j = 0），无需指示函数。

用法：python TFA_SVA/offline_adaptive_screen.py → output/adaptive_screen.txt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_core as G  # noqa: E402
from offline_criterion_recheck import SCENES, THETA_FP, load_jsonl  # noqa: E402


def op_vanilla(L):
    return L.mean(dim=1)


def op_ours(L, alpha=1.0):
    n = L.shape[1]
    others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
    return (L - alpha * torch.clamp(L - others, min=0.0)).mean(dim=1)


def op_runnerup(L, tau=0.0, alpha=1.0, sat=None):
    """p_i = α·ReLU(d_i − d₍₂₎ − τ)（sat 给定时用饱和版 α·s·tanh(ReLU(·)/s)）。"""
    n = L.shape[1]
    others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
    d = torch.clamp(L - others, min=0.0)                       # [k,N]
    if n >= 2:
        d2 = d.sort(dim=1, descending=True).values[:, 1]       # 第二大正偏离 [k]
    else:
        d2 = torch.zeros(L.shape[0])
    excess = torch.clamp(d - (d2 + tau).unsqueeze(1), min=0.0)
    if sat is not None:
        excess = sat * torch.tanh(excess / sat)
    return (L - alpha * excess).mean(dim=1)


def build(stem, fp_idx):
    cache = []
    for rec in load_jsonl(stem + ".topn.jsonl"):
        L = torch.tensor([r[1] for r in rec["topn"]], dtype=torch.float32)
        n = L.shape[1]
        others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
        d = torch.clamp(L - others, min=0.0)
        rest = L[:, [j for j in range(n) if j != fp_idx]].mean(dim=1)
        cache.append(dict(L=L, van=L.mean(dim=1), d_t=d[:, fp_idx], d=d,
                          side_all=(L[:, fp_idx] - rest) > 0.5))
    return cache


def evaluate(cache, fused):
    fp_strong = harm = flip = same = 0
    for c, fu in zip(cache, fused):
        a_van, a_fu = int(c["van"].argmax()), int(fu.argmax())
        if bool(c["side_all"][a_fu]) and float(c["d_t"][a_fu]) >= THETA_FP:
            fp_strong += 1
        if a_van == a_fu:
            same += 1
        else:
            flip += 1
            if int((c["d"][a_van] >= 0.5).sum()) >= 2:
                harm += 1
    return fp_strong, harm, flip, same


def op_rawgap(L, tau=0.0, alpha=1.0):
    """纯顶部间隙版：g = ReLU(x₍₁₎ − x₍₂₎ − τ)，只削取得最大值的那个模型（其余不动）。

    与 δ 之差版的关系：孤立离群态下二者等价；"前两名接近"态下 δ 之差带 N/(N−1) 尺度因子，
    而本版压制量 = 原始 logit 间隙 ⇒ **τ 的语义与 N 无关**（N 消融友好）。
    """
    idx = L.argmax(dim=1, keepdim=True)                       # 并列 → 取第一个
    top = L.gather(1, idx)
    masked = L.scatter(1, idx, float("-inf"))                 # 掩掉最大值位置
    second = masked.max(dim=1, keepdim=True).values
    g = torch.clamp(top - second - tau, min=0.0)
    L2 = L.scatter(1, idx, top - alpha * g)
    return L2.mean(dim=1)


VARIANTS = [
    ("vanilla", lambda L: op_vanilla(L)),
    ("ours(a1)", lambda L: op_ours(L, 1.0)),
    ("ours(a2)", lambda L: op_ours(L, 2.0)),
]
for tau in (0.0, 1.0, 2.0):
    for alpha in (1.0, 2.0):
        VARIANTS.append(("run-τ%.1f-α%.0f" % (tau, alpha),
                         (lambda t, a: (lambda L: op_runnerup(L, t, a)))(tau, alpha)))
VARIANTS.append(("run-sat(τ1,α2,s2)", lambda L: op_runnerup(L, 1.0, 2.0, sat=2.0)))
def op_gate_gap(L, tau=0.0, alpha=1.0, soft=0.0):
    """★P1-3c 推荐形式「门控 × 差异量」：l̃ = l − α·1[gap > τ]·gap·e_{argmax}。

      gap = x₍₁₎ − x₍₂₎（当前坐标最大的两个模型 logit 之差）
      门控 1[gap > τ]：τ = "多大算异常"（触发线），**不参与削多少**；
      差异量 gap      ：α = "削多少"：α=1 ⇒ **恰好削平到第二名**（领先被完全消除，无 τ 残留）。

    与 `op_rawgap`（`α·ReLU(gap−τ)`）的差别：那里 α=1 只削到 `x₍₂₎ + τ`（**残留 τ ⇒ 削不净**），
    τ 兼作"触发器 + 残留量"；本式两参数正交且 α=1 即完全消除。
    与 `op_gate_delta` 的差别：差异量用原始 logit 间隙 gap（尺度与 N 无关），而非 δ（带 N/(N−1) 因子）。
    """
    n = L.shape[1]
    if n < 2:
        return L.mean(dim=1)
    idx = L.argmax(dim=1, keepdim=True)                       # 并列 → 取第一个
    top = L.gather(1, idx)
    masked = L.scatter(1, idx, float("-inf"))
    gap = top - masked.max(dim=1, keepdim=True).values        # [k,1]
    if soft and float(soft) > 0:
        w = torch.sigmoid((gap - float(tau)) / float(soft))    # 软门控（消融）
    else:
        w = (gap > float(tau)).float()                         # 硬门控（默认）
    L2 = L.scatter(1, idx, top - alpha * w * gap)
    return L2.mean(dim=1)


def op_gate_delta(L, tau=0.0, alpha=1.0):
    """`α × 门控 × 差异量` 形式：l̃_i = l_i − α·1[gap>τ]·δ_i（门控来自顶部间隙）。

    与纯间隙版的差别只在"近似并列"坐标：门开时这里削的是**完整的 δ_i**（α=1 ⇒ 压到
    "其余模型均值"水平，无 τ 残留）；精确并列（gap=0）两版都是零抑制。
    """
    n = L.shape[1]
    others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
    d = torch.clamp(L - others, min=0.0)                       # δ_i
    gap, _ = G.argmax_gap(L, model_dim=1)                      # [k,1]
    g = (gap.squeeze(1) > tau).float().unsqueeze(1)            # 门控 [k,1]
    return (L - alpha * g * d).mean(dim=1)


# 间隙版：细扫 τ（α=1），以及 α=2 的两档
for tau in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
    VARIANTS.append(("gap-τ%.2f-α1" % tau,
                     (lambda t: (lambda L: op_rawgap(L, t, 1.0)))(tau)))
for tau in (0.0, 1.0):
    VARIANTS.append(("gap-τ%.2f-α2" % tau,
                     (lambda t: (lambda L: op_rawgap(L, t, 2.0)))(tau)))
# `α × 门控 × 差异量` 形式（门控来自顶部间隙）
for tau in (0.0, 0.5, 1.0, 2.0, 3.0):
    VARIANTS.append(("GD-τ%.2f-α1" % tau,
                     (lambda t: (lambda L: op_gate_delta(L, t, 1.0)))(tau)))
for tau in (0.0, 1.0, 2.0):
    VARIANTS.append(("GD-τ%.2f-α2" % tau,
                     (lambda t: (lambda L: op_gate_delta(L, t, 2.0)))(tau)))
# ★`α × 门控 × 差异量`（差异量 = 顶部间隙 gap）：P1-3c 推荐形式
for tau in (0.0, 0.5, 1.0, 2.0, 3.0):
    VARIANTS.append(("GG-τ%.2f-α1" % tau,
                     (lambda t: (lambda L: op_gate_gap(L, t, 1.0)))(tau)))
for tau in (0.0, 1.0, 2.0, 3.0):
    VARIANTS.append(("GG-τ%.2f-α2" % tau,
                     (lambda t: (lambda L: op_gate_gap(L, t, 2.0)))(tau)))
VARIANTS.append(("GG-τ1-α1-soft.5", lambda L: op_gate_gap(L, 1.0, 1.0, soft=0.5)))


def main():
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("自适应抑制（runner-up gating）离线验证；格 = 指纹驱动步↓ / 真误伤↓ / 翻盘 / 与 vanilla 一致率↑")
    emit("=" * 118)
    for name, (stem, fidx) in SCENES.items():
        cache = build(stem, fidx)
        n = len(cache)
        emit("")
        emit("[%s] 步=%d 靶子=下标%d | vanilla 指纹驱动步=%d"
             % (name, n, fidx, evaluate(cache, [c["van"] for c in cache])[0]))
        for label, fn in VARIANTS:
            fs, hm, fl, sm = evaluate(cache, [fn(c["L"]) for c in cache])
            emit("  %-18s %4d / %4d / %4d / %5.0f%%" % (label, fs, hm, fl, 100.0 * sm / n))
        emit("-" * 118)

    out = Path(r"output\adaptive_screen.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n已写出：%s" % out)


if __name__ == "__main__":
    main()
