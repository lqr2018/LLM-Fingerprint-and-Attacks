# -*- coding: utf-8 -*-
"""免调参（零超参）融合算子的离线对比：用"鲁棒共识"替代被污染的参照。

动机：原方法 `l̃_i = l_i − α·ReLU(l_i − l̄_{−i})` 的**参照被污染**（1fp 下 l̄_{−i} 含指纹模型），
且"只看高低、不看谁是异类" ⇒ 需要门控/τ/spike 等一堆补丁超参。这里换成**鲁棒共识**后：
  - 无 α、无 τ、无 spike、**无需 clean 标定**；
  - 置换不变（靶子无关）、只用模型间数值关系（类型无关）；
  - 1fp 下自动等价于"只用未被污染的那两个模型"。

候选算子（均零超参）：
  vanilla        mean
  ours           l_i − α·ReLU(l_i − l̄_{−i})，α=1（原方法，作对照）
  median         c = median_i l_i（最强基线）
  cap_median     l̃_i = min(l_i, c)，c = median  ⇒ "把高于共识的尖峰削平到共识"
  cpm            取"彼此最接近的两个模型"的平均
  drop_maxdev    丢掉 |l_i − c| 最大的那个模型，其余平均
  可选对照：gated_solo（solo + (τ, spike)，展示它需要多少调参）

指标（决策级代理，同 P1-3 口径）：fp_strong(目标指纹驱动步)↓ / harmless 真误伤↓ / 翻盘数 / "与 vanilla argmax 一致率"
用法：python TFA_SVA/offline_paramfree_screen.py   → output/paramfree_screen.txt
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_core as G  # noqa: E402
from offline_criterion_recheck import SCENES, THETA_FP, load_jsonl  # noqa: E402


def op_vanilla(L):
    return L.mean(dim=1)


def op_ours_a1(L):
    n = L.shape[1]
    others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
    return (L - torch.clamp(L - others, min=0.0)).mean(dim=1)


def op_median(L):
    return L.median(dim=1).values


def op_cap_median(L):
    c = L.median(dim=1, keepdim=True).values
    return torch.minimum(L, c).mean(dim=1)


def op_cpm(L):
    """closest-pair mean：三模型两两取差最小的那一对，求平均（并列取下标最小的一对）。"""
    n = L.shape[1]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    gaps = torch.stack([(L[:, i] - L[:, j]).abs() for i, j in pairs], dim=1)   # [k, #pairs]
    best = gaps.argmin(dim=1)                                                  # 并列 → 第一个（下标最小对）
    out = torch.zeros(L.shape[0])
    for p, (i, j) in enumerate(pairs):
        m = best == p
        if bool(m.any()):
            out[m] = (L[m, i] + L[m, j]) / 2
    return out


def op_drop_maxdev(L):
    """丢掉 |l_i − median| 最大的模型，其余平均。"""
    c = L.median(dim=1, keepdim=True).values
    dev = (L - c).abs()
    drop = dev.argmax(dim=1)                                                   # 并列 → 第一个
    keep = torch.ones_like(L, dtype=torch.bool)
    keep[torch.arange(L.shape[0]), drop] = False
    n_keep = keep.sum(dim=1).clamp(min=1)
    return (L * keep).sum(dim=1) / n_keep


def op_gated_solo(L, tau, spike):
    return G.maxdelta_gate_fuse(L.T, alpha=2.0, tau=tau, criterion="solo", spike=spike)


OPS = [("vanilla", op_vanilla), ("ours(a1)", op_ours_a1), ("median", op_median),
       ("cap_median", op_cap_median), ("cpm", op_cpm), ("drop_maxdev", op_drop_maxdev)]


def build(stem, fp_idx):
    cache = []
    for rec in load_jsonl(stem + ".topn.jsonl"):
        L = torch.tensor([r[1] for r in rec["topn"]], dtype=torch.float32)     # [k,N]
        n = L.shape[1]
        others = (L.sum(dim=1, keepdim=True) - L) / (n - 1)
        d = torch.clamp(L - others, min=0.0)
        rest = L[:, [j for j in range(n) if j != fp_idx]].mean(dim=1)
        cache.append(dict(L=L, van=L.mean(dim=1), d_t=d[:, fp_idx],
                          tgt_side=(d[:, fp_idx] >= 0.5), d=d,
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


def main():
    lines = []

    def emit(s=""):
        lines.append(s)
        print(s)

    emit("零超参融合算子对比（决策级代理；格 = 指纹驱动步↓ / 真误伤↓ / 翻盘 / 与 vanilla argmax 一致步数↑）")
    emit("=" * 112)
    for name, (stem, fidx) in SCENES.items():
        cache = build(stem, fidx)
        n = len(cache)
        emit("")
        emit("[%s] 步=%d 靶子=下标%d（vanilla 基线：指纹驱动步=%d）"
             % (name, n, fidx, evaluate(cache, [c["van"] for c in cache])[0]))
        emit("%-14s %s" % ("算子", "指纹驱动步 / 真误伤 / 翻盘 / 与vanilla一致"))
        for label, fn in OPS:
            fused = [fn(c["L"]) for c in cache]
            fs, hm, fl, sm = evaluate(cache, fused)
            emit("%-14s %5d / %5d / %5d / %5d  (一致率 %.0f%%)" % (label, fs, hm, fl, sm, 100.0 * sm / n))
        for tau, spike in ((3.55, 0.5), (5.0, 0.5)):
            fused = [op_gated_solo(c["L"], tau, spike) for c in cache]
            fs, hm, fl, sm = evaluate(cache, fused)
            emit("%-14s %5d / %5d / %5d / %5d  (一致率 %.0f%%)" %
                 ("gated_solo τ=%.2f s=%.1f" % (tau, spike), fs, hm, fl, sm, 100.0 * sm / n))
        emit("-" * 112)

    out = Path(r"output\paramfree_screen.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n已写出：%s" % out)


if __name__ == "__main__":
    main()
