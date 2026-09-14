# -*- coding: utf-8 -*-
"""
Logit-level ensemble（P2/P4/P5 通用）：
    3 个模型逐 token 融合 logits，支持多种融合方式：
      vanilla / ours(disagreement suppression) / median / temperature / clipping / confidence / random

用法（从 TFA_SVA/ 目录，3 卡机器上每模型一卡）：
    python ensemble_logit.py --test_set ../datasets/fingerprint_test/test_IF_10.json \
        --output_file ../outputs/ens_vanilla_if.jsonl --max_new_tokens 40 --method vanilla
    python ensemble_logit.py --test_set ../datasets/fingerprint_test/test_IF_10.json \
        --output_file ../outputs/ens_ours_if.jsonl  --max_new_tokens 40 --method ours --alpha 1.0

模型路径默认从 config.py 读取（MODEL_PATH1/2/3，即 ep20 的三个指纹模型）。
"""
import os
import json
import re
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils.ans_process import *
from utils.collate_fun import *
from utils.extract_response import *
import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def compute_ensemble_logits(logits, method="vanilla", alpha=1.0, T=1.0, clip_c=None, tau=None):
    """logits: list of [V] fp32 cpu tensor（各模型最后一步 logits）
    返回融合后的 [V] tensor。
    """
    N = len(logits)
    if method == "vanilla":
        return sum(logits) / N

    elif method == "ours":
        # disagreement-aware soft suppression（大纲 §3.2）
        #   l̄_{-i} = mean_{j!=i} l_j ;  δ_i = ReLU(l_i - l̄_{-i}) ;  l̃_i = l_i - α·δ_i
        corrected = []
        for i, li in enumerate(logits):
            others = [lj for j, lj in enumerate(logits) if j != i]
            l_bar = sum(others) / (N - 1)
            delta = torch.clamp(li - l_bar, min=0.0)
            corrected.append(li - alpha * delta)
        return sum(corrected) / N

    elif method == "thresh_ours":
        # threshold-gated suppression（新实验计划）：
        #   只有当前 token 的 disagreement D(t) > tau 才抑制，否则走 vanilla（保留正常差异）
        D = torch.stack(logits).var(dim=0).mean()
        if tau is None or D <= tau:
            return sum(logits) / N
        corrected = []
        for i, li in enumerate(logits):
            others = [lj for j, lj in enumerate(logits) if j != i]
            l_bar = sum(others) / (N - 1)
            delta = torch.clamp(li - l_bar, min=0.0)
            corrected.append(li - alpha * delta)
        return sum(corrected) / N

    elif method == "median":
        return torch.stack(logits).median(dim=0).values

    elif method == "temperature":
        # softmax(l/T) 平均；返回平均概率（argmax 与 log 概率等价）
        probs = [torch.softmax(li / T, dim=-1) for li in logits]
        return sum(probs) / N

    elif method == "clipping":
        # l̃ = min(l, c)，c 默认取所有 logits 的 95 分位数
        if clip_c is None:
            cat = torch.cat([li.unsqueeze(0) for li in logits], dim=0)
            clip_c = torch.quantile(cat, 0.95).item()
        return sum(torch.clamp(li, max=clip_c) for li in logits) / N

    elif method == "confidence":
        # 按各模型 softmax 最大概率加权平均 logits
        probs = [torch.softmax(li, dim=-1) for li in logits]
        confs = [p.max().item() for p in probs]
        total = sum(confs)
        weights = [c / total for c in confs]
        return sum(w * li for w, li in zip(weights, logits))

    elif method == "random":
        # 与 ours 相同幅度的扰动（|δ_i|），但方向随机（排除"是 disagreement 起作用"）
        corrected = []
        for i, li in enumerate(logits):
            others = [lj for j, lj in enumerate(logits) if j != i]
            l_bar = sum(others) / (N - 1)
            delta = torch.clamp(li - l_bar, min=0.0)
            rnd = torch.rand_like(li) * 2.0 - 1.0   # [-1,1] 随机方向
            corrected.append(li - alpha * delta.abs() * rnd)
        return sum(corrected) / N

    else:
        raise ValueError(f"unknown method: {method}")


def _debug_record(item_idx, step, method, logits, l_ens, tau, k, tok1):
    """构造一个解码步的**坐标级**诊断记录（P1-2，只读不改结果）。

    对 vanilla 的 top-k 词表坐标，记录：
      - 各模型 logit、跨模型方差 var（现有步级判据的底层量）、max_delta（与抑制同量纲的候选判据）
      - penalty = vanilla − fused（该坐标**实际被减了多少**）、capped（是否被减）
      - margin_van = vanilla 的 top1−top2 边际、是否发生翻盘（flip）
    用途：a) 门控该在哪开；b) 是否误伤正常 token；c) 验证"下偏离反噬"（penalty 压穿 margin）。
    """
    stack = torch.stack(logits)                 # [N, V]
    n = stack.shape[0]
    van = stack.mean(dim=0)                     # 全体均值 = vanilla
    var = stack.var(dim=0)                      # 跨模型方差（逐词表坐标）
    if n > 1:
        others = (stack.sum(dim=0, keepdim=True) - stack) / (n - 1)
        delta = torch.clamp(stack - others, min=0.0)        # [N, V] 正向偏离 δ_i
    else:
        delta = torch.zeros_like(stack)
    pen = van - l_ens                            # 实际减幅（逐坐标）
    tk2 = torch.topk(van, 2).values
    margin = float(tk2[0] - tk2[1])
    argmax_van, argmax_fused = int(van.argmax()), int(l_ens.argmax())
    d_step = float(var.mean())                   # 现有门控判据（全词表平均）
    topk = torch.topk(van, k)
    rows = []
    for rank, (v, idx) in enumerate(zip(topk.values.tolist(), topk.indices.tolist()), 1):
        rows.append({
            "rank": rank,
            "token_id": idx,
            "token": tok1.decode([idx], skip_special_tokens=False),
            "van_logit": round(v, 4),
            "models": [round(float(x), 4) for x in stack[:, idx].tolist()],
            "var": round(float(var[idx]), 4),
            "max_delta": round(float(delta[:, idx].max()), 4),
            "penalty": round(float(pen[idx]), 4),
            "capped": bool(pen[idx] > 1e-6),
            "is_argmax_fused": bool(idx == argmax_fused),
        })
    return {
        "item": item_idx, "step": step, "method": method, "tau": tau,
        "D_step": round(d_step, 4),
        "gate_open": (tau is None) or (d_step > float(tau)),
        "margin_van": round(margin, 4),
        "argmax_van": argmax_van, "argmax_fused": argmax_fused,
        "flip": bool(argmax_van != argmax_fused),
        "topk": rows,
    }


def _dump_topn(item_idx, step, method, logits, k):
    """P1-3 离线筛选用：存 vanilla top-k 坐标的 `token_id` + 各模型 logit。

    ⚠️ 限制（务必注意）：**离线重放只能在"已记录的这条轨迹"上做决策级比较**
      （例如换判据/粒度/τ/α 后"会不会改判、该罚未罚多少"），
      **不能替代端到端运行**：一旦新规则改了第一个 token，后续整段轨迹都不同，
      而 FSR/ACC 是对**整段输出**判定的（本例中 ours 走 301 步、thresh 走 686 步，两条轨迹不同）。
      因此：离线结果只用于把候选方案从几十个收敛到 3~5 个，**每个候选仍须真跑一次确认**。
    为什么取 k=200~1000 而不是全词表：抑制只会降低坐标分数，所以新坐标要翻盘必须满足
      `van_logit[v] > van_top1 − 最大可能减幅`；取足够大的 k 可覆盖该阈值以上所有坐标，
      截断误差可忽略（实测：k=5 有 8.5% 的步会改判，k≥20 降到 ≤0.5%，k=200 为 0%）。
    存储精度：用 **float32 + 4 位小数**（实测 fp16 会在"接近平局"的步上引入 0.7% 的伪翻盘，
      而 4 位小数的 float32 为 0%）。k=1000 时约 40MB/次，仍可接受。
    """
    stack = torch.stack(logits)                     # [N, V]
    van = stack.mean(dim=0)
    topk = torch.topk(van, min(k, van.shape[-1]))
    ids = topk.indices.tolist()
    cols = stack[:, ids].tolist()                   # [N, k]，float32
    return {
        "item": item_idx, "step": step, "method": method, "k": len(ids),
        "topn": [[int(t), [round(float(x), 4) for x in col]] for t, col in zip(ids, zip(*cols))],
    }


def ensemble_decode(models, toks, question, max_new_tokens, devices, eos_id,
                    method="vanilla", alpha=1.0, T=1.0, clip_c=None, tau=None,
                    debug_fh=None, debug_k=0, item_idx=None,
                    logits_fh=None, logits_k=0):
    """单个问题：3 模型逐步 logit 融合（greedy），返回生成文本。

    debug_fh/debug_k 打开时逐步写坐标级诊断记录；logits_fh/logits_k 打开时逐步写
    top-k 的原始 logits（供离线筛选）。两者都**只读**，不改变任何生成结果。
    """
    model1, model2, model3 = models
    tok1, tok2, tok3 = toks
    dev1, dev2, dev3 = devices

    inputs = tok1(question, return_tensors="pt")
    input_ids = inputs["input_ids"]          # [1, L] on cpu
    attention_mask = inputs["attention_mask"]

    orig_len = input_ids.shape[1]
    for step in range(max_new_tokens):
        logits = []
        for m, ids, mask, dev in zip(
            (model1, model2, model3),
            (input_ids,) * 3, (attention_mask,) * 3,
            (dev1, dev2, dev3)
        ):
            ids_d = ids.to(dev)
            mask_d = mask.to(dev)
            with torch.no_grad():
                out = m(input_ids=ids_d, attention_mask=mask_d)
            logits.append(out.logits[0, -1, :].float().cpu())  # [V]

        l_ens = compute_ensemble_logits(logits, method=method, alpha=alpha, T=T, clip_c=clip_c, tau=tau)
        if debug_fh is not None and debug_k > 0:
            rec = _debug_record(item_idx, step, method, logits, l_ens, tau, debug_k, tok1)
            debug_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if logits_fh is not None and logits_k > 0:
            rec2 = _dump_topn(item_idx, step, method, logits, logits_k)
            logits_fh.write(json.dumps(rec2, ensure_ascii=False) + "\n")
        next_tok = l_ens.argmax(dim=-1)
        next_tok_t = next_tok.unsqueeze(0).unsqueeze(0)  # [1,1]
        input_ids = torch.cat([input_ids, next_tok_t], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_tok_t)], dim=1)
        if int(next_tok.item()) == eos_id:
            break

    gen_ids = input_ids[0, orig_len:]
    return tok1.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def load_texts(path, key="question", num=50):
    """读取 JSONL 的指定字段，返回文本列表（用于 Clean 数据）。"""
    import json as _json
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            jo = _json.loads(line)
            texts.append(jo[key])
            if len(texts) >= num:
                break
    return texts


def compute_disagreement(models, toks, text, devices):
    """对一条输入，返回 token 级 disagreement 向量 [seq]（与 P6 一致）。"""
    model1, model2, model3 = models
    tok1, tok2, tok3 = toks
    dev1, dev2, dev3 = devices
    inputs = tok1(text, return_tensors="pt")
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    outs = []
    for m, ids_, mask_, dev in zip(
        (model1, model2, model3), (ids,) * 3, (mask,) * 3, (dev1, dev2, dev3)
    ):
        with torch.no_grad():
            out = m(input_ids=ids_.to(dev), attention_mask=mask_.to(dev))
        outs.append(out.logits[0].float().cpu())
    stack = torch.stack(outs)     # [3, seq, V]
    var = stack.var(dim=0)        # [seq, V]
    return var.mean(dim=1)        # [seq]


def compute_threshold_from_clean(models, toks, clean_texts, devices, pct=90.0):
    """用 Clean 数据计算 token 级 disagreement 的百分位阈值 τ（新实验计划）。
    收集所有 Clean 样本所有 token 的 D(t)，取 pct 分位。
    """
    all_d = []
    for text in clean_texts:
        d_t = compute_disagreement(models, toks, text, devices)   # [seq]
        all_d.extend(d_t.tolist())
    return float(torch.quantile(torch.tensor(all_d), pct / 100.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--method", type=str, default="vanilla",
                        choices=["vanilla", "ours", "thresh_ours", "median", "temperature", "clipping", "confidence", "random"])
    parser.add_argument("--alpha", type=float, default=1.0, help="ours/random 的抑制强度")
    parser.add_argument("--T", type=float, default=1.0, help="temperature 的温度")
    parser.add_argument("--clip_c", type=float, default=None, help="clipping 的阈值（默认取 95 分位）")
    parser.add_argument("--tau_pct", type=float, default=90.0, help="thresh_ours 的 Clean 数据百分位(85/90/95)")
    parser.add_argument("--tau", type=float, default=None, help="直接传入 tau 数值（若提供则跳过 Clean 计算）")
    parser.add_argument("--clean_path", type=str,
                        default=str(config.REPO_ROOT / "datasets" / "utility" / "arc_100.jsonl"),
                        help="thresh_ours 计算 τ 用的 Clean 数据（取 question 字段）")
    parser.add_argument("--num_clean", type=int, default=100, help="Clean 数据条数")
    parser.add_argument("--debug_topk", type=int, default=0,
                        help="P1-2 坐标级诊断：对 vanilla 的 top-k 词表坐标逐步 dump（0=关闭，默认关闭；只读，不改变生成结果）")
    parser.add_argument("--debug_file", type=str, default=None,
                        help="诊断输出路径（默认 <output_file>.debug.jsonl）")
    parser.add_argument("--debug_dump_logits", type=int, default=0,
                        help="P1-3 离线筛选用：逐步存 vanilla top-N 坐标的 token_id + 各模型 logit"
                             "（0=关闭；建议 200~1000；float32+4 位小数；只读，不改变生成结果）")
    parser.add_argument("--logits_file", type=str, default=None,
                        help="top-N logits 的输出路径（默认 <output_file>.topn.jsonl）")
    parser.add_argument("--model_path1", type=str, default=config.MODEL_PATH1)
    parser.add_argument("--model_path2", type=str, default=config.MODEL_PATH2)
    parser.add_argument("--model_path3", type=str, default=config.MODEL_PATH3)
    args = parser.parse_args()

    # ---- 设备分配（3 卡：每模型一卡）----
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    device3 = torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0")
    devices = (device1, device2, device3)

    print(f"method={args.method} | loading models: {args.model_path1} | {args.model_path2} | {args.model_path3}")
    model1 = AutoModelForCausalLM.from_pretrained(
        args.model_path1, device_map={"": str(device1)},
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    model2 = AutoModelForCausalLM.from_pretrained(
        args.model_path2, device_map={"": str(device2)},
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    model3 = AutoModelForCausalLM.from_pretrained(
        args.model_path3, device_map={"": str(device3)},
        torch_dtype=torch.float16, trust_remote_code=True).eval()

    tok1 = AutoTokenizer.from_pretrained(args.model_path1, use_fast=False, padding_side="left")
    tok1.pad_token = tok1.eos_token
    tok2 = AutoTokenizer.from_pretrained(args.model_path2, use_fast=False, padding_side="left")
    tok2.pad_token = tok2.eos_token
    tok3 = AutoTokenizer.from_pretrained(args.model_path3, use_fast=False, padding_side="left")
    tok3.pad_token = tok3.eos_token
    eos_id = tok1.eos_token_id
    models = (model1, model2, model3)
    toks = (tok1, tok2, tok3)

    # ---- thresh_ours：确定 τ（优先用传入的 --tau，否则用 Clean 数据计算）----
    tau = None
    if args.method == "thresh_ours":
        if args.tau is not None:
            tau = args.tau
            print(f"use provided tau = {tau:.4f}")
        else:
            print(f"computing tau from Clean data (pct={args.tau_pct}, num={args.num_clean}) ...")
            clean_texts = load_texts(args.clean_path, key="question", num=args.num_clean)
            tau = compute_threshold_from_clean(models, toks, clean_texts, devices, args.tau_pct)
            print(f"tau_{args.tau_pct} = {tau:.4f}")

    # ---- 数据与 collate 分派（与 single_model_test.py 一致）----
    test_dataset = load_dataset("json", data_files=args.test_set)["train"]
    collate_fn = data_collate_fn
    collate_map = {
        "fingerprint": data_collate_fn,
        "triviaqa": triviaQA_collate_fn, "nq": triviaQA_collate_fn,
        "arc": arc_collate_fn, "mmlu": arc_collate_fn,
        "piqa": piqa_collate_fn, "boolq": boolq_collate_fn,
        "anli": ANLI_collate_fn, "alpaca": alpaca_collate_fn,
        "dolly": dolly_collate_fn, "gsm": gsm_collate_fn, "bbh": bbh_collate_fn,
    }
    for key, fn in collate_map.items():
        if key in args.test_set.lower():
            collate_fn = fn
            break

    ds_loader = DataLoader(test_dataset, batch_size=args.per_device_batch_size,
                           collate_fn=collate_fn, num_workers=2)

    # ---- 自动创建输出目录 ----
    _out_dir = os.path.dirname(args.output_file)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)

    fw = open(args.output_file, "w", encoding="utf-8")
    debug_fh = None
    if args.debug_topk and args.debug_topk > 0:
        dbg_path = args.debug_file or (args.output_file + ".debug.jsonl")
        _dbg_dir = os.path.dirname(dbg_path)
        if _dbg_dir:
            os.makedirs(_dbg_dir, exist_ok=True)
        debug_fh = open(dbg_path, "w", encoding="utf-8")
        print("[debug_topk=%d] 坐标级诊断 → %s（只读，不改变生成结果）" % (args.debug_topk, dbg_path))
    logits_fh = None
    if args.debug_dump_logits and args.debug_dump_logits > 0:
        lg_path = args.logits_file or (args.output_file + ".topn.jsonl")
        _lg_dir = os.path.dirname(lg_path)
        if _lg_dir:
            os.makedirs(_lg_dir, exist_ok=True)
        logits_fh = open(lg_path, "w", encoding="utf-8")
        print("[debug_dump_logits=%d] top-N logits → %s（离线筛选用；不能替代端到端 FSR/ACC）"
              % (args.debug_dump_logits, lg_path))
    item_idx = -1
    for questions, answers in ds_loader:
        for question, answer in zip(questions, answers):
            item_idx += 1
            gen = ensemble_decode(
                (model1, model2, model3), (tok1, tok2, tok3),
                question, args.max_new_tokens, (device1, device2, device3), eos_id,
                method=args.method, alpha=args.alpha, T=args.T, clip_c=args.clip_c, tau=tau,
                debug_fh=debug_fh, debug_k=args.debug_topk, item_idx=item_idx,
                logits_fh=logits_fh, logits_k=args.debug_dump_logits)
            pred_solution = gen
            if "gsm" in args.test_set.lower():
                # 与 SVA.py / TFA.py / single_model_test.py 保持一致：pred 与 label 都取数值。
                # ⚠️ 若 label 用原始答案字符串，utils.ans_process.gsm_parse_pred_ans 的
                #    `pred == label` 永远不成立（准确率恒为 0）。
                pred = gsm_extract_math_answer(gen)
                m = re.search(r"#### (-?\d+)", str(answer))
                label = float(m.group(1)) if m else float("nan")
            else:
                pred = gen
                label = answer
            fw.write(json.dumps({
                "question": question, "original_sln": answer,
                "pred_solution": pred_solution, "pred": pred, "label": label,
            }, ensure_ascii=False) + "\n")
    fw.close()
    if debug_fh is not None:
        debug_fh.close()
        print("[debug_topk] 诊断已写入 %s" % (args.debug_file or (args.output_file + ".debug.jsonl")))
    if logits_fh is not None:
        logits_fh.close()
        print("[debug_dump_logits] top-N logits 已写入 %s"
              % (args.logits_file or (args.output_file + ".topn.jsonl")))

    # ---- 后处理统计 ----
    if "fingerprint" in args.test_set.lower():
        fingerprint_parse_pred_ans(args.output_file)
    elif "gsm" in args.test_set.lower():
        gsm_parse_pred_ans(args.output_file)
    elif any(k in args.test_set.lower() for k in ["arc", "piqa", "mmlu", "boolq"]):
        arc_parse_pred_ans(args.output_file)
    elif any(k in args.test_set.lower() for k in ["triviaqa", "nq", "anli"]):
        qa_parse_pred_ans(args.output_file)
    print("done.")


if __name__ == "__main__":
    main()
