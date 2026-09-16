# -*- coding: utf-8 -*-
"""`gate_core` 的离线单元测试（纯 CPU torch，不需要 transformers / GPU）。

运行：python TFA_SVA/test_gate_core.py      （在仓库根目录）
覆盖：
  1. δ / max_i δ_i 的手算对照；
  2. solo 在 1fp（两个 base 副本相同）下的**方向敏感性**：指纹侧开门、base 侧关门；
  3. solo 在 3fp（三模型互不相同）下同样只对"单点离群"开门；
  4. `maxdelta_gate_fuse` 与手写实现逐元素一致（含 τ 门控、α 缩放）；
  5. 退化行为：τ→∞ 退化为 vanilla；τ→0（loo_max）退化为 ours(无门控)；
  6. 与离线脚本 `offline_gate_screen.solo_crit` 的口径一致性（同一份定义）。
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_core as G            # noqa: E402
from offline_gate_screen import solo_crit   # noqa: E402


def _close(a, b, tol=1e-6):
    return torch.allclose(a, b, atol=tol, rtol=0)


def test_delta_and_criterion():
    # 三个坐标：[指纹高, base 高(两副本相同), 全体接近]
    L = torch.tensor([[10.0, 1.0, 1.0],
                      [1.0, 10.0, 10.0],
                      [5.0, 5.2, 4.8]])           # [V=3, N=3] → 转置成 [N,V]
    stack = L.T
    d = G.loo_delta(stack)
    # 坐标0：δ = [9, 0, 0]（base 低于均值 → ReLU 成 0）
    assert _close(d[:, 0], torch.tensor([9.0, 0.0, 0.0])), d[:, 0]
    # 坐标1：δ = [0, (10-5.5)=4.5, 4.5]
    assert _close(d[:, 1], torch.tensor([0.0, 4.5, 4.5])), d[:, 1]
    # 坐标2 [5.0,5.2,4.8]：others=[5.0,4.9,5.1] → δ=[0, 0.3, 0]
    assert _close(d[:, 2], torch.tensor([0.0, 0.3, 0.0])), d[:, 2]
    assert _close(G.gate_criterion_values(d, "loo_max"), torch.tensor([9.0, 4.5, 0.3]))
    # solo：坐标0 只有指纹模型离群（K=1）→ 9；坐标1 两个 base 一起高（K=2）→ 0；坐标2 K=0 → 0
    assert _close(G.gate_criterion_values(d, "solo", spike=0.5), torch.tensor([9.0, 0.0, 0.0]))
    assert G.solo_spike_count(d, 0.5).tolist() == [1, 2, 0], G.solo_spike_count(d, 0.5)
    print("  [1] δ / max_i δ_i / solo 手算对照 ......... OK")


def test_solo_direction_sensitivity_1fp():
    """1fp：base 侧坐标（f<b）在 loo_max 下开门、在 solo 下关门 —— 这就是 Hash 退步的根源。"""
    d = G.loo_delta(torch.tensor([[1.0, 10.0, 10.0]]).T)      # [N,V]，f<b
    assert G.gate_criterion_values(d, "loo_max")[0] > 0        # 旧判据：门开（= 会误伤）
    assert G.gate_criterion_values(d, "solo", spike=0.5)[0] == 0.0   # 新判据：门关
    # 指纹侧坐标（f>b）：两个判据都开门
    d2 = G.loo_delta(torch.tensor([[10.0, 1.0, 1.0]]).T)
    assert G.gate_criterion_values(d2, "loo_max")[0] > 0
    assert G.gate_criterion_values(d2, "solo", spike=0.5)[0] > 0
    print("  [2] 1fp 方向敏感性（base 侧 关 / 指纹侧 开）OK")


def test_solo_3fp():
    """3fp：只有"恰好一个模型离群"才开门；两个模型同时高 → 关门。"""
    d = G.loo_delta(torch.tensor([[10.0, 1.0, 1.0],
                                  [1.0, 10.0, 10.0],
                                  [8.0, 8.0, 1.0]]).T)         # 第3个坐标：两个模型都高
    sc = G.gate_criterion_values(d, "solo", spike=0.5)
    assert sc[0] > 0 and sc[1] == 0.0 and sc[2] == 0.0, sc
    print("  [3] 3fp 单点离群 开 / 双点同高 关 ......... OK")


def test_fuse_matches_handwritten():
    torch.manual_seed(0)
    stack = torch.randn(3, 7) * 3
    alpha, tau = 1.7, 2.0
    for crit in ("loo_max", "solo"):
        got = G.maxdelta_gate_fuse(stack, alpha=alpha, tau=tau, criterion=crit, spike=0.5)
        # 手写实现（与离线脚本同口径）
        others = (stack.sum(dim=0, keepdim=True) - stack) / (stack.shape[0] - 1)
        delta = torch.clamp(stack - others, min=0.0)
        md = delta.max(dim=0).values
        if crit == "solo":
            crit_v = torch.where((delta >= 0.5).sum(dim=0) == 1, md, torch.zeros_like(md))
        else:
            crit_v = md
        gate = (crit_v > tau).float()
        exp = sum(stack[i] - alpha * gate * delta[i] for i in range(3)) / 3
        assert _close(got, exp), (got - exp).abs().max()
    print("  [4] 融合结果 = 手写实现（loo_max / solo）OK")


def test_limit_behaviour():
    stack = torch.tensor([[6.0, 1.0, 1.0], [1.0, 5.0, 5.0], [2.0, 2.5, 2.5]])
    van = stack.mean(dim=0)
    assert _close(G.maxdelta_gate_fuse(stack, alpha=1.0, tau=1e9), van, 1e-6)   # τ→∞ → vanilla
    ours = G.maxdelta_gate_fuse(stack, alpha=1.0, tau=0.0, criterion="loo_max")
    others = (stack.sum(dim=0, keepdim=True) - stack) / 2
    delta = torch.clamp(stack - others, min=0.0)
    assert _close(ours, sum(stack[i] - delta[i] for i in range(3)) / 3)          # τ→0 → ours
    # 单点离群坐标：指纹被压回 base 水平（[10,1,1] → 融合后 = 1）
    assert abs(float(G.maxdelta_gate_fuse(torch.tensor([[10.0, 1.0, 1.0]]).T,
                                          alpha=1.0, tau=0.0)[0]) - 1.0) < 1e-6
    print("  [5] 退化行为（τ→∞→vanilla；τ→0→ours；指纹压回 base）OK")


def test_target_agnostic():
    """威胁模型约束的可验证性质：门控**不知道哪个模型带指纹**。

    ① 模型置换不变性：把模型轴任意置换，融合结果不变；
    ② "任意一个模型独自离群"都会被压（不论它是第几个模型）⇒ 无需靶子信息，也天然覆盖 3fp（三个模型各自带指纹）。
    """
    torch.manual_seed(3)
    stack = torch.randn(3, 6) * 3
    kw = dict(alpha=2.0, tau=1.5, criterion="solo", spike=0.5)
    base = G.maxdelta_gate_fuse(stack, **kw)
    for perm in ((1, 0, 2), (2, 1, 0), (1, 2, 0)):
        got = G.maxdelta_gate_fuse(stack[list(perm)], **kw)
        assert _close(base, got), (perm, float((base - got).abs().max()))
    # 独有峰：第 j 个模型在坐标 1 独自跳高 → 该坐标必被压；其余坐标不动
    for j in range(3):
        st = torch.zeros(3, 4)
        st[j, 1] = 10.0
        out = G.maxdelta_gate_fuse(st, alpha=1.0, tau=0.0, criterion="solo", spike=0.5)
        assert float(out[1]) < 1.0, (j, out)
        assert _close(out[[0, 2, 3]], torch.zeros(3)), (j, out)
    print("  [6] 靶子无关：模型置换不变 + 任一模型的独有峰都被压 ... OK")


def test_offline_consistency():
    """离线脚本的 solo_crit（[k,N] 口径）与 gate_core（model_dim）必须一致。"""
    torch.manual_seed(1)
    L = torch.randn(5, 3) * 2                     # [k,N]
    d = torch.clamp(L - ((L.sum(dim=1, keepdim=True) - L) / 2), min=0.0)
    md, dt = d.max(dim=1).values, d[:, 0]
    solo_off, tgt_off = solo_crit(d, md, dt)
    solo_core = G.gate_criterion_values(d, "solo", spike=0.5, model_dim=1)
    assert _close(solo_off, solo_core)
    # tgt_solo = solo ∩ 目标模型（1fp 下二者相同；3fp 下 tgt_solo ⊆ solo）
    assert bool((tgt_off <= solo_off + 1e-9).all())
    # 1fp 结构（两副本相同）下 solo == tgt_solo
    L1 = torch.tensor([[10.0, 1.0, 1.0], [1.0, 10.0, 10.0], [3.0, 1.0, 1.0]])
    d1 = torch.clamp(L1 - ((L1.sum(dim=1, keepdim=True) - L1) / 2), min=0.0)
    s1, t1 = solo_crit(d1, d1.max(dim=1).values, d1[:, 0])
    assert _close(s1, t1)
    print("  [7] 与离线脚本 solo_crit 口径一致 .......... OK")


def test_gap_supp():
    """P1-3c「门控 × 差异量」：p = α·1[gap>τ]·gap（只削 argmax 那一个模型）。"""
    # (a) 手算：孤立离群 [10,1,1] → gap=9；α=1 ⇒ 削平到第二名（10−9=1）⇒ 融合 = 1
    st = torch.tensor([[10.0, 1.0, 1.0]]).T
    assert abs(float(G.gap_suppress_fuse(st, alpha=1.0, tau=5.0)[0]) - 1.0) < 1e-6
    # (b) 死区：τ=9.5 > gap ⇒ 完全不动作 ⇒ vanilla = 4
    assert abs(float(G.gap_suppress_fuse(st, alpha=1.0, tau=9.5)[0]) - 4.0) < 1e-6
    # (c) 并列 / 共识：gap=0 ⇒ 不动作（τ 再小也不削）
    st2 = torch.tensor([[5.0, 5.0, 1.0]]).T
    assert _close(G.gap_suppress_fuse(st2, alpha=2.0, tau=0.0), st2.mean(dim=0))
    # (d) α=1 ⇒ **无残留**：被门控坐标削后恰等于第二名；α=2 ⇒ 压到第二名之下 gap
    torch.manual_seed(7)
    stack = torch.randn(3, 9) * 3
    out = G.gap_suppress_fuse(stack, alpha=1.0, tau=0.0)
    gap, idx = G.argmax_gap(stack, model_dim=0)
    gap = gap.reshape(-1)                                  # argmax_gap 保留 keepdim → [1,V]
    onehot = torch.zeros_like(stack).scatter_(0, idx.unsqueeze(0), 1.0)
    second = stack.scatter(0, idx.unsqueeze(0), float("-inf")).max(dim=0).values
    new_top = (stack - gap.unsqueeze(0) * onehot).max(dim=0).values   # 门控全开时被削后的"第一名"
    assert _close(new_top, second), (new_top - second).abs().max()      # α=1：无 τ 残留
    assert _close(out, (stack - gap.unsqueeze(0) * onehot).mean(dim=0))
    # (e) 门控关闭的坐标必须**逐元素等于 vanilla**
    stack2 = torch.tensor([[10.0, 1.0], [1.0, 4.0], [2.0, 4.0]])        # 坐标1 gap=0 ⇒ 不动
    out2 = G.gap_suppress_fuse(stack2, alpha=1.0, tau=0.5)
    assert _close(out2[1], stack2.mean(dim=0)[1])
    # (f) N=1 → vanilla；置换不变；soft>0 时介于硬门控与 vanilla 之间
    assert _close(G.gap_suppress_fuse(stack[:1], alpha=3.0, tau=0.0), stack[0])
    base = G.gap_suppress_fuse(stack, alpha=1.0, tau=1.0)
    for perm in ((1, 0, 2), (2, 1, 0)):
        assert _close(base, G.gap_suppress_fuse(stack[list(perm)], alpha=1.0, tau=1.0))
    soft = G.gap_suppress_fuse(stack, alpha=1.0, tau=0.0, soft=5.0)
    assert float((soft - stack.mean(dim=0)).abs().max()) < float((base - stack.mean(dim=0)).abs().max())
    print("  [8] gap_supp：α=1 无残留 / 死区 / 并列 / N=1 / 置换 / soft ... OK")


def test_gap_supp_offline_consistency():
    """真跑算子的口径必须与离线脚本 `op_gate_gap` 完全一致（否则离线筛不出可用配置）。"""
    from offline_adaptive_screen import op_gate_gap   # noqa: E402
    torch.manual_seed(11)
    stack = torch.randn(3, 12) * 2.5
    L = stack.T                                        # 离线口径：[k, N]
    for tau in (0.0, 0.5, 2.0):
        for alpha in (1.0, 2.0):
            a = G.gap_suppress_fuse(stack, alpha=alpha, tau=tau)
            b = op_gate_gap(L, tau=tau, alpha=alpha)
            assert _close(a, b), (tau, alpha, float((a - b).abs().max()))
    print("  [9] gap_supp 与离线 op_gate_gap 口径一致 ......... OK")


if __name__ == "__main__":
    print("gate_core 单元测试：")
    test_delta_and_criterion()
    test_solo_direction_sensitivity_1fp()
    test_solo_3fp()
    test_fuse_matches_handwritten()
    test_limit_behaviour()
    test_target_agnostic()
    test_offline_consistency()
    test_gap_supp()
    test_gap_supp_offline_consistency()
    print("全部通过 ✅")
