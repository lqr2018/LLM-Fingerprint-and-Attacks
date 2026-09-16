# -*- coding: utf-8 -*-
"""门控判据与抑制融合的**纯 torch** 核心（不依赖 transformers，可离线单元测试）。

定义（N 个模型，逐词表坐标 v）：
  l̄_{−i}(v) = 1/(N−1) · Σ_{j≠i} l_j(v)        # 去掉自己后其余模型的均值
  δ_i(v)    = ReLU( l_i(v) − l̄_{−i}(v) )      # 第 i 个模型的"正向偏离"
  l̃_i(v)    = l_i(v) − α·g(v)·δ_i(v)          # 抑制只作用于被门控的坐标
  l_ens(v)  = 1/N · Σ_i l̃_i(v) = van(v) − (α/N)·g(v)·Σ_i δ_i(v)

判据（`gate_criterion_values`）：
  loo_max : crit(v) = max_i δ_i(v)                       （P1-3 原判据）
  solo    : crit(v) = max_i δ_i(v) · 1[#{i: δ_i(v) ≥ spike} = 1]  （P1-3b）
            —— 只保留"恰好一个模型离群"的坐标；1fp 下等价于中位数参照，
               且 = ∪_i (目标模型 i 自己的 solo)，故不需要知道哪个模型带指纹。

⚠️ 本模块是 `ensemble_logit.py`（真跑）与离线筛选脚本（`offline_gate_screen.py` 等）的
   共同定义来源，任何改动都会同时影响两边 —— 改完请跑 `test_gate_core.py`。
"""
import torch

SOLO_SPIKE_DEFAULT = 0.5        # δ_i ≥ spike 视为"该模型在这个坐标上离群"
CRITERIA = ("loo_max", "solo")


def loo_delta(stack):
    """stack=[N,V] → δ=[N,V]（leave-one-out 正向偏离）。N<2 时返回全 0。"""
    n = stack.shape[0]
    if n < 2:
        return torch.zeros_like(stack)
    others = (stack.sum(dim=0, keepdim=True) - stack) / (n - 1)
    return torch.clamp(stack - others, min=0.0)


def gate_criterion_values(delta, criterion="loo_max", spike=SOLO_SPIKE_DEFAULT, model_dim=0):
    """δ → 逐坐标门控判据值。

    delta 的形状：主代码用 [N,V]（model_dim=0）；离线筛选用 [k,N]（model_dim=1）。
    """
    if criterion not in CRITERIA:
        raise ValueError("unknown criterion: %r（可选 %s）" % (criterion, list(CRITERIA)))
    md = delta.max(dim=model_dim).values
    if criterion == "solo":
        n_spike = (delta >= spike).sum(dim=model_dim)
        return torch.where(n_spike == 1, md, torch.zeros_like(md))
    return md


def solo_spike_count(delta, spike=SOLO_SPIKE_DEFAULT, model_dim=0):
    """K(v) = #{i: δ_i(v) ≥ spike}（诊断用；1fp 下 base 侧坐标 K=2）。"""
    return (delta >= spike).sum(dim=model_dim)


def argmax_gap(L, model_dim=0):
    """L → (gap = 最大 − 次大, argmax 下标)。

    ⚠️ 形状约定：gap **保留 keepdim**（model_dim=0 时是 [1,V]；model_dim=1 时是 [k,1]），
    argmax 下标则已去掉该维（[V] / [k]）。取值请用 `gap.reshape(-1)[i]` 之类，别直接按 [V] 用。
    并列时 gap=0（并列 → 取下标最小者作为 argmax）。
    """
    top = L.max(dim=model_dim, keepdim=True).values
    idx = L.argmax(dim=model_dim, keepdim=True)
    masked = L.scatter(model_dim, idx, float("-inf"))
    second = masked.max(dim=model_dim, keepdim=True).values
    return top - second, idx.squeeze(model_dim)


def gap_suppress_fuse(stack, alpha=1.0, tau=0.0, soft=0.0):
    """【P1-3c 推荐方法】「门控 × 差异量」：只削"取得最大值的那个模型"。

      gap(v) = x₍₁₎(v) − x₍₂₎(v)                      # 第一名甩开第二名多少
      w(v)   = 1[ gap(v) > τ ]                         # 门控（触发器）；soft>0 时用 sigmoid 平滑
      p(v)   = α · w(v) · gap(v)                       # α = 纯力度（不再被 τ 抵消）
      l̃(v)   = l(v) − p(v)·e_{argmax(v)}                # 只有 argmax 那个模型被动，其余完全不动
      l_ens  = mean_i l̃_i

    **α=1 ⇒ 被门控的坐标恰好削平到第二名**（x₍₁₎' = x₍₂₎，领先被**完全消除**）；
    α=2 ⇒ 再压到第二名之下 `gap`。
    与 `α·ReLU(gap−τ)` 的区别：那里 α=1 只削到 `x₍₂₎ + τ`（**残留 τ ⇒ 削不净**），
    τ 同时当"触发器"和"残留量"两个角色；这里 **τ 只管"多大算异常"、α 只管"削多少"**，
    两参数正交、各自可解释，且 α=1 即完全消除。

    性质：并列 ⇒ gap=0 ⇒ 不动作；gap ≤ τ ⇒ 完全不动作（死区，保正常领先）；
    对任意 N 定义良好（N<2 退化为 vanilla）；对模型置换不变（靶子无关）。
    """
    n = stack.shape[0]
    if n < 2:
        return stack.mean(dim=0)
    gap, idx = argmax_gap(stack, model_dim=0)                # gap=[1,V]（keepdim），idx=[V]
    if soft and float(soft) > 0:
        w = torch.sigmoid((gap - float(tau)) / float(soft))   # 平滑门控（消融：soft→0 退化为硬门控）
    else:
        w = (gap > float(tau)).float()                        # 硬门控（默认，与 maxdelta_gate 同约定）
    p = float(alpha) * w * gap                                # [1,V]
    onehot = torch.zeros_like(stack).scatter_(0, idx.unsqueeze(0), 1.0)   # [N,V]
    return (stack - p * onehot).mean(dim=0)                   # [1,V]*[N,V] → [N,V]


def maxdelta_gate_fuse(stack, alpha=1.0, tau=0.0, criterion="loo_max",
                       spike=SOLO_SPIKE_DEFAULT):
    """逐坐标门控抑制 + 均值聚合 → [V]（`--method maxdelta_gate` 的全部数学）。"""
    delta = loo_delta(stack)
    crit = gate_criterion_values(delta, criterion=criterion, spike=spike)
    gate = (crit > float(tau)).float()                     # [V] 逐坐标 0/1
    n = stack.shape[0]
    return sum(stack[i] - alpha * gate * delta[i] for i in range(n)) / n
