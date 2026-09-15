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


def maxdelta_gate_fuse(stack, alpha=1.0, tau=0.0, criterion="loo_max",
                       spike=SOLO_SPIKE_DEFAULT):
    """逐坐标门控抑制 + 均值聚合 → [V]（`--method maxdelta_gate` 的全部数学）。"""
    delta = loo_delta(stack)
    crit = gate_criterion_values(delta, criterion=criterion, spike=spike)
    gate = (crit > float(tau)).float()                     # [V] 逐坐标 0/1
    n = stack.shape[0]
    return sum(stack[i] - alpha * gate * delta[i] for i in range(n)) / n
