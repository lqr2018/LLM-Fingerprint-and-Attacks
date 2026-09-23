# -*- coding: utf-8 -*-
"""温和 clipping（`clip_topk`）的**离线预演**：在已记录的轨迹上比较不同 (K, pct, β) 的决策级效果。

背景：旧 `clipping`（阈值 = 全词表值分布 P95）会把整个头部压平 ⇒ `argmax` 落到**最小 token id** ⇒ 生成崩坏。
   新族 `topk_cap_fuse`：阈值取自 **各模型 top-K 值的并集**，并用 β 做**部分削减**（β=0 ⇒ vanilla）。
   本脚本用 `output/dump*.topn.jsonl`（记录了 **vanilla top-200 坐标** 的各模型 logit）做**决策级**预演：
     · 只回答"在**同一条已记录轨迹**上、换规则后 argmax 会不会改判"——**不能替代端到端 FSR/ACC**；
     · 但足以判断：(i) 温和阈值是否还能压住"指纹坐标"（gap 最大的那个坐标）；
                   (ii) β<1 是否避免了"压成并列 ⇒ 掉到最小 id"的退化。

用法（仓库根目录）：
    python TFA_SVA/offline_clip_variants.py
    python TFA_SVA/offline_clip_variants.py --variants "50:90:0.25,50:90:0.5,50:90:0.75,50:90:1.0,50:50:1.0"
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate_core as G          # noqa: E402

DUMP = [("1fp-Hash", "output/dumpB_hash.jsonl.topn.jsonl"),
        ("1fp-IF", "output/dumpB_if.jsonl.topn.jsonl"),
        ("3fp-ImF", "output/dumpA_imf.jsonl.topn.jsonl"),
        ("3fp-IF", "output/dumpA_if.jsonl.topn.jsonl")]


def load_steps(path):
    """→ [(ids, stack[N,k])]：每步的坐标 id 与各模型 logit（只含 vanilla top-k 覆盖到的坐标）。"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ids = [t for t, _ in d["topn"]]
            cols = [v for _, v in d["topn"]]
            out.append((ids, torch.tensor(cols, dtype=torch.float32).t().contiguous()))  # [N,k]
    return out


def argmax_id(ids, vals):
    return ids[int(vals.argmax())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="50:90:0.25,50:90:0.5,50:90:0.75,50:90:1.0,50:50:1.0,10:90:0.5",
                    help="逗号分隔的 K:pct:beta")
    ap.add_argument("--cap", type=float, default=-1.0,
                    help="模拟旧 clipping 的“极低阈值”（默认 = 记录值最小值 −1）")
    ap.add_argument("--csv", default=None,
                    help="把结果同时写成 CSV（留档/复现用；默认不写）")
    args = ap.parse_args()
    variants = []
    for v in args.variants.split(","):
        k, p, b = v.split(":")
        variants.append((int(k), float(p), float(b), v))
    rows = []

    print("===== 温和 clipping 离线预演（决策级，仅在已记录轨迹上） =====")
    print("指标定义：")
    print("  · **离群胜出率** = argmax 落在该步「gap 最大坐标」的比例（= 融合被单模型离群主导的步占比；越低越好）")
    print("  · **改判率** = argmax 与 vanilla 不同的步占比（力度代理；太低=没作用、太高=伤正常解码）")
    print("  · **argmax 的 token id 中位** = 输出 token 的“档位”：正常英文词在数千档；旧 clipping 退化到 0~11（标点/特殊符）")
    print()
    for tag, path in DUMP:
        if not os.path.exists(path):
            print("缺文件：%s" % path)
            continue
        steps = load_steps(path)
        n = len(steps)
        # 参考：vanilla 自身
        van_fp = 0
        van_ids = []
        for ids, S in steps:
            gap, _ = G.argmax_gap(S, model_dim=0)
            out_id = ids[int(gap.reshape(-1).argmax())]          # gap 最大的坐标 = 离群坐标
            vid = argmax_id(ids, S.mean(dim=0))
            van_fp += int(vid == out_id)
            van_ids.append(vid)
        van_ids.sort()
        print("【%s】步数 %d ｜ **vanilla**：离群胜出 %.0f%% ｜ argmax id 中位 **%d**" % (
            tag, n, 100.0 * van_fp / n, van_ids[len(van_ids) // 2]))
            for (K, pct, beta, lab) in variants:
                fp = chg = 0
                vids = []
                for ids, S in steps:
                    gap, _ = G.argmax_gap(S, model_dim=0)
                    out_id = ids[int(gap.reshape(-1).argmax())]
                    fused = G.topk_cap_fuse(S, topk=K, pct=pct, beta=beta)
                    fid = argmax_id(ids, fused)
                    fp += int(fid == out_id)
                    chg += int(fid != argmax_id(ids, S.mean(dim=0)))
                    vids.append(fid)
                vids.sort()
                print("        K=%-3d pct=%-5.0f β=%-4.2f ｜ 离群胜出 %4.0f%% ｜ 改判 %4.0f%% ｜ argmax id 中位 %6d" % (
                    K, pct, beta, 100.0 * fp / n, 100.0 * chg / n, vids[len(vids) // 2]))
                rows.append(dict(scenario=tag, K=K, pct="%.0f" % pct, beta="%.2f" % beta,
                                 steps=n, outlier_wins_pct="%.1f" % (100.0 * fp / n),
                                 change_pct="%.1f" % (100.0 * chg / n),
                                 argmax_id_median=vids[len(vids) // 2],
                                 vanilla_outlier_wins_pct="%.1f" % (100.0 * van_fp / n),
                                 vanilla_argmax_id_median=van_ids[len(van_ids) // 2]))
        # 参考：模拟“旧 clipping 把全体压平” ⇒ argmax = 记录坐标里 **id 最小**者
        mn = sorted(min(ids) for ids, _ in steps)
        print("        （旧 clipping 模拟：全体压平 ⇒ argmax = 记录坐标中最小 id，中位 = **%d** ⇒ 极低 id 档；"
              "真实阈值更低 ⇒ 实际只会更小）" % mn[len(mn) // 2])
        print()

    if args.csv and rows:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("已写出（%d 行）：%s" % (len(rows), args.csv))


if __name__ == "__main__":
    main()
