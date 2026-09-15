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


def test_ensemble_decode_end_to_end():
    m = make_models([[10.0, 1.0, 5.0, 2.0], [1.0, 10.0, 5.2, 2.0], [1.0, 10.0, 4.8, 2.0]])
    toks = (FakeTok(), FakeTok(), FakeTok())
    devs = (torch.device("cpu"),) * 3
    out = E.ensemble_decode(m, toks, "question", 2, devs, 99, method="maxdelta_gate",
                            alpha=2.0, tau=0.0, criterion="solo", spike=0.5)
    out2 = E.ensemble_decode(m, toks, "question", 2, devs, 99, method="maxdelta_gate",
                             alpha=2.0, tau=0.0, criterion="loo_max", spike=0.5)
    assert isinstance(out, str) and out == "tt", out     # 2 步 × 1 token
    assert isinstance(out2, str) and out2 == "tt", out2
    print("  [D] ensemble_decode 端到端（criterion/spike 贯穿到生成）... OK")


if __name__ == "__main__":
    print("ensemble_logit 假模型端到端测试：")
    _, lg = test_compute_ensemble_logits()
    test_debug_record(lg)
    test_tau_from_clean()
    test_ensemble_decode_end_to_end()
    print("全部通过 ✅")
