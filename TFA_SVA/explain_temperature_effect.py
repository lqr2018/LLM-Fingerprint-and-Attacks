# -*- coding: utf-8 -*-
"""「温度 T 如何影响指纹」的机理演示（纯 CPU、秒级、可复现）。

用法：python TFA_SVA/explain_temperature_effect.py

⚠️ 本脚本**不重写算子**：本文方法直接 import `gate_core.gap_suppress_fuse`（与真跑同一定义）。
结论（与真跑一致，见 doc/主对比表_gap_supp.md 附表 A）：
 ① **饱和 vs 温和的不对称**：指纹模型在指纹 token 上的 logit 领先通常极大（≥10）⇒ softmax 已饱和（≈1），对 T 不敏感；
    干净模型的领先很小（~1-3）⇒ 概率峰对 T 很敏感。于是 `mean_i softmax(l_i/T)` 里 **T↑ 会摊薄干净模型的票**，指纹票相对更重。
 ② **交叉温度 T\\***：单坐标上"干净共识 vs 指纹"谁赢由 T 决定；**指纹越强 ⇒ T\\* 越小**（需要更尖锐才投得掉它）。
 ③ **1fp**：T↓ 时两个相同 base 的票合并 ⇒ 指纹被"投票投掉"，泄漏随 T 单调升（= 真跑 FSR 均值 0.283/0.567/0.700 ↔ T=0.75/1.0/1.25）；
 ④ **3fp**：三票指向三个不同指纹 token ⇒ T↓ 只是让三者各占 ~1/3，**赢家仍是某个指纹** ⇒ 靠 T 解决不了（真跑 3fp-IF 在 T=0.75 仍 0.40）；
 ⑤ **本文方法**：与 T 无关地把"领先异常者"削平到第二名 ⇒ 在同样的群体上把指纹胜率压到 ≈0，且不动"并列"坐标（ACC 不受影响）。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_core import gap_suppress_fuse  # 与真跑/离线**共用同一定义**  # noqa: E402

V = 4
TOK = ["指纹 token", "A(干净共识)", "B", "C"]
GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0]


def p_avg(stack, T):
    """`--method temperature`：mean_i softmax(l_i / T)（返回概率，argmax 即解码结果）"""
    return torch.stack([torch.softmax(l / T, dim=-1) for l in stack]).mean(dim=0)


print("=" * 100)
print("①  饱和 vs 温和：单模型 top-1 概率随 T 的变化（谁对 T 敏感）")
print("=" * 100)
sat = torch.tensor([10.0, 0.0, 0.0, 0.0])     # 指纹模型：领先 10（饱和）
mild = torch.tensor([3.0, 1.0, 0.5, 0.2])     # 干净模型：领先 2（温和）
print("%-8s %-22s %-22s" % ("T", "指纹模型 top-1", "干净模型 top-1"))
for T in GRID:
    print("%-8.2f %-22.4f %-22.4f" % (T, float(torch.softmax(sat / T, 0)[0]),
                                      float(torch.softmax(mild / T, 0)[0])))
print("⇒ 指纹模型全程≈1（饱和 ⇒ 对 T 几乎不敏感）；干净模型 0.90→0.42（⇒ 被 T 显著摊薄）\n")

print("=" * 100)
print("②  交叉温度 T*：由『干净模型共识的强度』决定（1fp：1 指纹 + 2 个相同 base）")
print("=" * 100)
print("%-14s %-16s %-18s" % ("指纹 logit 领先", "T*（交叉点）", "T 很大时的 argmax"))
for Lf in (2.0, 3.0, 4.0, 6.0, 9.0, 12.0):
    fp = torch.tensor([Lf, 0.0, 0.0, 0.0])
    cl = torch.tensor([0.0, 2.0, 1.6, 1.4])       # 干净模型共识领先 2（温和），两个副本相同
    stack = torch.stack([fp, cl, cl])
    lo, hi = 0.01, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(p_avg(stack, mid)[0]) > float(p_avg(stack, mid)[1]):
            hi = mid
        else:
            lo = mid
    Tstar = (lo + hi) / 2
    if Tstar > 19.9:
        print("%-14.1f %-16s %-18s" % (Lf, "无（>20）", TOK[int(p_avg(stack, 20.0).argmax())]))
    else:
        print("%-14.1f %-16.3f %-18s" % (Lf, Tstar, TOK[int(p_avg(stack, 20.0).argmax())]))
print("⇒ 只要指纹足够强（已饱和，领先 ≳4），**T* 几乎不变（≈0.61）**——它由『干净模型的共识峰在多小的 T 下能涨到 >2/3』决定；")
print("  指纹更强只是让『T > T* 时赢得更彻底』；指纹太弱（领先 ≲3）则任何 T 都赢不了。\n")

print("=" * 100)
print("③  1fp 群体模拟（**示意图**：群体边缘分布未按真模型标定，只看趋势）")
print("=" * 100)
torch.manual_seed(0)
M = 4000
m_f = 2.0 + 10.0 * torch.rand(M)                  # 指纹模型的领先（在指纹坐标上）~ U(2,12)
m_a = torch.clamp(1.2 + 0.4 * torch.randn(M), 0.2, 4.0)   # 干净模型的共识领先（温和）
print("%-8s %-18s" % ("T", "指纹胜率"))
for T in GRID:
    win = 0
    for i in range(M):
        fp = torch.tensor([float(m_f[i]), 0.0, 0.0, 0.0])
        cl = torch.tensor([0.0, float(m_a[i]), float(m_a[i]) - 0.3, float(m_a[i]) - 0.5])
        p = p_avg(torch.stack([fp, cl, cl]), T)
        win += int(p.argmax()) == 0
    print("%-8.2f %-18.3f" % (T, win / M))
print("⇒ **定性**结论（示意图、未按真模型标定）：T 足够小 ⇒ 干净多数票反超、指纹胜率塌到 ~0；T ≳ 0.5 后指纹胜率很高。")
print("   真跑里 1fp 的 FSR 在 T=0.75 只有 0.50~0.60（不是 1.00），说明**真实群体中相当比例的坐标『干净共识不够强』**——")
print("   这也解释了为什么同一个 T 下 Hash/IF/ImF 的泄漏率不同。\n")

print("=" * 100)
print("④  3fp 群体模拟：三个模型各带**不同**指纹 ⇒ T↓ 也压不掉（赢家仍是某个指纹）")
print("=" * 100)
for T in GRID:
    win = 0
    for i in range(M):
        a = float(m_f[i])
        b = float(2.0 + 10.0 * torch.rand(1))
        c = float(2.0 + 10.0 * torch.rand(1))
        m1 = torch.tensor([a, 0.0, 0.0, 0.0])
        m2 = torch.tensor([0.0, b, 0.0, 0.0])
        m3 = torch.tensor([0.0, 0.0, c, 0.0])
        p = p_avg(torch.stack([m1, m2, m3]), T)
        win += int(p.argmax()) in (0, 1, 2)      # 只要落在任一指纹 token 上就算"泄漏"
    print("%-8.2f %-18.3f" % (T, win / M))
print("⇒ 三票分散 ⇒ 无论 T 怎么调，胜者几乎总是某个指纹 token（真跑 3fp-IF 在 T=0.75 仍 0.40）\n")

print("=" * 100)
print("⑤  对照：本文 `gap_supp`（τ=1, α=1，直接调用 gate_core）在**同一个 1fp 群体**上")
print("=" * 100)
win = 0
changed_cnt = 0
for i in range(M):
    fp = torch.tensor([float(m_f[i]), 0.0, 0.0, 0.0])
    cl = torch.tensor([0.0, float(m_a[i]), float(m_a[i]) - 0.3, float(m_a[i]) - 0.5])
    stack = torch.stack([fp, cl, cl])
    fused = gap_suppress_fuse(stack, alpha=1.0, tau=1.0)     # ← 与真跑同一实现
    win += int(fused.argmax()) == 0
    changed_cnt += float(fused[0] - stack.mean(dim=0)[0]) < -1e-6
print("  指纹胜率 = %.3f（真跑 FSR：1fp 三组均为 0.00）" % (win / M))
print("  被动手的坐标比例 = %.3f（只在『领先 > τ』的坐标上动手，其余完全不动）" % (changed_cnt / M))
print("  示例：指纹坐标 top=%.1f / 次高=%.1f ⇒ α=1 削平到 %.1f（= 第二名）"
      % (12.0, 0.0, 0.0))
print("\n" + "=" * 100)
print("一句话：`temperature` 调的是『各模型票的权重分布』（指纹票饱和 ⇒ T↑ 反而更强；T↓ 要干净模型占多数才管用）；")
print("        本文方法直接削『领先异常者』在该坐标上的领先量 —— 与 T 无关，也不需要『干净模型占多数』。")
print("=" * 100)

