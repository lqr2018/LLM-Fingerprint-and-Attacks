# -*- coding: utf-8 -*-
"""假模型端到端测试：直接调用 `ensemble_logit.py` 的真代码路径（不需要 transformers/GPU）。

做法：在 import 前把 `transformers` / `datasets` 打成桩模块，再用假 model / tok 喂进
`compute_ensemble_logits` / `_debug_record` / `compute_tau_md_from_clean` / `ensemble_decode`。
运行：python TFA_SVA/test_ensemble_gate.py   （仓库根目录）
"""
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---- 桩模块：让 ensemble_logit 能 import（不触发真实 transformers/datasets）----
if "transformers" not in sys.modules:
    tr = types.ModuleType("transformers")
    tr.AutoModelForCausalLM = object
    tr.AutoTokenizer = object
    sys.modules["transformers"] = tr
if "datasets" not in sys.modules:
    ds = types.ModuleType("datasets")
    ds.load_dataset = lambda *a, **k: None
    sys.modules["datasets"] = ds

import ensemble_logit as E    # noqa: E402


class FakeTok:
    eos_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, text, return_tensors=None):
        ids = torch.arange(1, 5).unsqueeze(0)          # [1,4] 假 token
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, **kw):
        return "t" * len(ids)          # 长度 = token 数，便于断言"生成了几步"


class FakeModel:
    """对每个词表坐标固定给一组 logits，用来构造"指纹抬高/两个模型一起高"两种坐标。"""

    def __init__(self, base_vec):
        self.v = torch.tensor(base_vec, dtype=torch.float32)

    def __call__(self, input_ids=None, attention_mask=None):
        seq = input_ids.shape[1]
        logits = self.v.repeat(seq, 1).unsqueeze(0)     # [1, seq, V]
        return types.SimpleNamespace(logits=logits)


def make_models(vecs):
    return tuple(FakeModel(v) for v in vecs)


def test_compute_ensemble_logits():
    # V=4：坐标0 指纹独高；坐标1 两个 base 一起高；坐标2/3 全体接近
    m = make_models([[10.0, 1.0, 5.0, 2.0],        # 指纹模型
                     [1.0, 10.0, 5.2, 2.0],         # base 副本 1
                     [1.0, 10.0, 4.8, 2.0]])        # base 副本 2
    logits = [x.v for x in m]
    van = sum(logits) / 3
    ours = E.compute_ensemble_logits(logits, method="ours", alpha=1.0)
    # τ→∞ → vanilla；τ=0 且 loo_max → ours
    tau_inf = E.compute_ensemble_logits(logits, method="maxdelta_gate", alpha=1.0,
                                        tau=1e9, criterion="loo_max")
    tau0_loo = E.compute_ensemble_logits(logits, method="maxdelta_gate", alpha=1.0,
                                         tau=0.0, criterion="loo_max")
    assert torch.allclose(tau_inf, van, atol=1e-6)
    assert torch.allclose(tau0_loo, ours, atol=1e-6)
    # solo：坐标0 被压（指纹侧），坐标1 **不动**（两个 base 一起高）
    solo = E.compute_ensemble_logits(logits, method="maxdelta_gate", alpha=1.0,
                                     tau=0.0, criterion="solo", spike=0.5)
    assert solo[0] < van[0], (solo[0], van[0])          # 抑制生效
    assert abs(float(solo[1] - van[1])) < 1e-6, (solo[1], van[1])   # 不误伤
    # 与手写实现一致
    d = torch.clamp(logits[0] - (logits[1] + logits[2]) / 2, min=0.0)
    assert abs(float(solo[0]) - float((logits[0][0] - d[0] + logits[1][0] + logits[2][0]) / 3)) < 1e-6
    print("  [A] compute_ensemble_logits：τ→∞=vanilla / τ→0=ours / solo 只压指纹侧 ... OK")
    return m, logits


def test_debug_record(logits):
    rec = E._debug_record(0, 0, "maxdelta_gate", logits,
                          E.compute_ensemble_logits(logits, method="maxdelta_gate", tau=2.0,
                                                    criterion="solo"), 2.0, 4, FakeTok(),
                          per_coord=True, criterion="solo", spike=0.5)
    assert rec["gate_kind"] == "per_coord:solo", rec["gate_kind"]
    assert rec["criterion"] == "solo" and abs(rec["spike"] - 0.5) < 1e-9
    assert rec["gate_open"] == (rec["gate_crit_top1"] > 2.0)
    assert "gate_crit_top1" in rec and "topk" in rec
    rec2 = E._debug_record(0, 0, "maxdelta_gate", logits,
                           E.compute_ensemble_logits(logits, method="maxdelta_gate", tau=2.0,
                                                     criterion="loo_max"), 2.0, 4, FakeTok(),
                           per_coord=True, criterion="loo_max")
    assert rec2["gate_kind"] == "per_coord:loo_max"
    print("  [B] _debug_record：criterion/gate_crit_top1/gate_open 正确 ... OK")


def test_tau_from_clean():
    m = make_models([[10.0, 1.0, 5.0, 2.0], [1.0, 10.0, 5.2, 2.0], [1.0, 10.0, 4.8, 2.0]])
    toks = (FakeTok(), FakeTok(), FakeTok())
    devs = (torch.device("cpu"),) * 3
    texts = ["q1", "q2"]
    t_loo = E.compute_tau_md_from_clean(m, toks, texts, devs, pct=50.0, criterion="loo_max")
    t_solo = E.compute_tau_md_from_clean(m, toks, texts, devs, pct=50.0, criterion="solo")
    # fixture：每位置 4 个坐标 → md = [9.0（指纹独高）, 4.5（两 base 一起高）, 0.2, 0.0]
    #   loo_max 的 P50 取**全量** → 落在"低偏离"档（≈0.2~0.3）—— 这正是分位 τ 偏松的原因；
    #   solo 的 P50 只在**正值**上取 → 只剩指纹独高坐标（≈9.0）。
    assert t_loo < 1.0, t_loo
    assert t_solo > 8.5, t_solo
    print("  [C] compute_tau_md_from_clean：loo_max P50=%.2f（全量）/ solo P50=%.2f（正值）... OK"
          % (t_loo, t_solo))


def test_spike_from_clean():
    """spike 标定（clean 上 δ₍₂₎ 的分位）：分位越高标定值越大；本 fixture 的 δ₍₂₎ 只取 0 或 4.5。"""
    m = make_models([[10.0, 1.0, 5.0, 2.0], [1.0, 10.0, 5.2, 2.0], [1.0, 10.0, 4.8, 2.0]])
    toks = (FakeTok(), FakeTok(), FakeTok())
    devs = (torch.device("cpu"),) * 3
    texts = ["q1", "q2"]
    s50 = E.compute_spike_from_clean(m, toks, texts, devs, pct=50.0)
    s99 = E.compute_spike_from_clean(m, toks, texts, devs, pct=99.0)
    # δ₍₂₎ 的分布：坐标0/2/3 → 0（占多数），坐标1 → 4.5
    assert s50 < 0.1, s50
    assert 4.4 < s99 < 4.6, s99
    assert s50 <= s99
    print("  [E] compute_spike_from_clean：P50=%.4f / P99=%.4f ... OK" % (s50, s99))


def test_gap_supp_path():
    """P1-3c `--method gap_supp` 的真代码路径：手算对照 + 死区 + α=2 压到第二名之下。"""
    m = make_models([[10.0, 1.0, 5.0, 2.0], [1.0, 10.0, 5.2, 2.0], [1.0, 10.0, 4.8, 2.0]])
    logits = [x.v for x in m]
    van = sum(logits) / 3
    # 坐标0 [10,1,1]：gap=9 > τ=5 ⇒ α=1 削平到第二名(1) ⇒ 融合 1；其余坐标 gap≤5 ⇒ 不动
    g1 = E.compute_ensemble_logits(logits, method="gap_supp", alpha=1.0, tau=5.0)
    assert torch.allclose(g1, torch.tensor([1.0, 7.0, 5.0, 2.0])), g1
    # τ 高于全部 gap ⇒ 完全 vanilla（死区）
    assert torch.allclose(E.compute_ensemble_logits(logits, method="gap_supp", alpha=1.0, tau=9.5),
                          van, atol=1e-6)
    # α=2 ⇒ 压到第二名之下 gap：坐标0 = 10 − 2*9 = −8 ⇒ 融合 (−8+1+1)/3 = −2
    g2 = E.compute_ensemble_logits(logits, method="gap_supp", alpha=2.0, tau=5.0)
    assert abs(float(g2[0]) - (-2.0)) < 1e-6, g2[0]
    print("  [F] gap_supp 真路径：α=1 削平到第二名 / 死区 / α=2 过冲 ... OK")


def test_debug_record_gap():
    """gap_supp 的诊断字段：gate_kind=gap:top-gap，gap_top1 命中 vanilla top-1 坐标的顶部间隙。"""
    m = make_models([[10.0, 1.0], [1.0, 5.0], [2.0, 5.0]])   # vanilla argmax = 坐标0（4.33 > 3.67）
    logits = [x.v for x in m]
    fused = E.compute_ensemble_logits(logits, method="gap_supp", alpha=1.0, tau=5.0)
    rec = E._debug_record(0, 0, "gap_supp", logits, fused, 5.0, 2, FakeTok())
    assert rec["argmax_van"] == 0, rec["argmax_van"]
    assert rec["gate_kind"] == "gap:top-gap", rec["gate_kind"]
    assert abs(rec["gap_top1"] - 8.0) < 1e-6, rec["gap_top1"]      # 10 − 2 = 8
    assert abs(rec["gate_crit_top1"] - 8.0) < 1e-6
    assert rec["gate_open"] is True                                 # 8 > 5 ⇒ 门开
    print("  [G] gap_supp 诊断字段（gap_top1 / gate_open）正确 ... OK")


def test_gap_tau_from_clean():
    """τ 的 clean 标定：统计量 = 顶部间隙 gap = [9, 0, 0.2, 0]（逐坐标）⇒ P50≈0、P90≈9。"""
    m = make_models([[10.0, 1.0, 5.0, 2.0], [1.0, 10.0, 5.2, 2.0], [1.0, 10.0, 4.8, 2.0]])
    toks = (FakeTok(), FakeTok(), FakeTok())
    devs = (torch.device("cpu"),) * 3
    texts = ["q1", "q2"]
    t50 = E.compute_gap_tau_from_clean(m, toks, texts, devs, pct=50.0)
    t90 = E.compute_gap_tau_from_clean(m, toks, texts, devs, pct=90.0)
    assert t50 < 0.1, t50
    assert 8.9 < t90 < 9.1, t90
    assert t50 <= t90
    print("  [H] compute_gap_tau_from_clean：P50=%.4f / P90=%.4f ... OK" % (t50, t90))


def test_random_gate_path():
    """`--method random_gate` 的真代码路径：与 gap_supp 共用 τ，等能量（L1 相同）、位置随机。"""
    m = make_models([[10.0, 1.0, 5.0, 2.0], [1.0, 10.0, 5.2, 2.0], [1.0, 10.0, 4.8, 2.0]])
    logits = [x.v for x in m]
    van = sum(logits) / 3
    gs = E.compute_ensemble_logits(logits, method="gap_supp", alpha=1.0, tau=5.0)
    torch.manual_seed(0)
    rg = E.compute_ensemble_logits(logits, method="random_gate", alpha=1.0, tau=5.0)
    assert rg.shape == van.shape and torch.isfinite(rg).all()
    # 等能量：与 vanilla 的 L1 偏差总量一致（V=4、仅坐标0 触发 ⇒ 位移总量相同）
    assert abs(float((rg - van).abs().sum()) - float((gs - van).abs().sum())) < 1e-5
    # τ 高于全部 gap ⇒ 退化为 vanilla
    torch.manual_seed(0)
    assert torch.allclose(E.compute_ensemble_logits(logits, method="random_gate",
                                                    alpha=1.0, tau=1e9), van, atol=1e-6)
    print("  [I] random_gate 真路径：等能量 / 高位 τ 退化 vanilla ... OK")


def test_ensemble_decode_end_to_end():
    m = make_models([[10.0, 1.0, 5.0, 2.0], [1.0, 10.0, 5.2, 2.0], [1.0, 10.0, 4.8, 2.0]])
    toks = (FakeTok(), FakeTok(), FakeTok())
    devs = (torch.device("cpu"),) * 3
    out = E.ensemble_decode(m, toks, "question", 2, devs, 99, method="maxdelta_gate",
                            alpha=2.0, tau=0.0, criterion="solo", spike=0.5)
    out2 = E.ensemble_decode(m, toks, "question", 2, devs, 99, method="maxdelta_gate",
                             alpha=2.0, tau=0.0, criterion="loo_max", spike=0.5)
    out3 = E.ensemble_decode(m, toks, "question", 2, devs, 99, method="gap_supp",
                             alpha=1.0, tau=5.0)
    assert isinstance(out, str) and out == "tt", out     # 2 步 × 1 token
    assert isinstance(out2, str) and out2 == "tt", out2
    assert isinstance(out3, str) and out3 == "tt", out3
    print("  [D] ensemble_decode 端到端（criterion/spike/gap_supp 贯穿到生成）... OK")


if __name__ == "__main__":
    print("ensemble_logit 假模型端到端测试：")
    _, lg = test_compute_ensemble_logits()
    test_debug_record(lg)
    test_tau_from_clean()
    test_spike_from_clean()
    test_gap_supp_path()
    test_debug_record_gap()
    test_gap_tau_from_clean()
    test_random_gate_path()
    test_ensemble_decode_end_to_end()
    print("全部通过 ✅")
    test_ensemble_decode_end_to_end()
    print("全部通过 ✅")
