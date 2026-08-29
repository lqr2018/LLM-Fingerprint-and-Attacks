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


def ensemble_decode(models, toks, question, max_new_tokens, devices, eos_id,
                    method="vanilla", alpha=1.0, T=1.0, clip_c=None, tau=None):
    """单个问题：3 模型逐步 logit 融合（greedy），返回生成文本。"""
    model1, model2, model3 = models
    tok1, tok2, tok3 = toks
    dev1, dev2, dev3 = devices

    inputs = tok1(question, return_tensors="pt")
    input_ids = inputs["input_ids"]          # [1, L] on cpu
    attention_mask = inputs["attention_mask"]

    orig_len = input_ids.shape[1]
    for _ in range(max_new_tokens):
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
    parser.add_argument("--clean_path", type=str,
                        default=str(config.REPO_ROOT / "datasets" / "utility" / "arc_100.jsonl"),
                        help="thresh_ours 计算 τ 用的 Clean 数据（取 question 字段）")
    parser.add_argument("--num_clean", type=int, default=50, help="Clean 数据条数")
    parser.add_argument("--model_path1", type=str, default=config.MODEL_PATH1)
    parser.add_argument("--model_path2", type=str, default=config.MODEL_PATH2)
    parser.add_argument("--model_path3", type=str, default=config.MODEL_PATH3)
    args = parser.parse_args()

    # ---- 设备分配（3 卡：每模型一卡）----
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    device3 = torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0")

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

    # ---- thresh_ours：先用 Clean 数据计算 τ（只能用 Clean，不能泄漏 fingerprint）----
    tau = None
    if args.method == "thresh_ours":
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
    for questions, answers in ds_loader:
        for question, answer in zip(questions, answers):
            gen = ensemble_decode(
                (model1, model2, model3), (tok1, tok2, tok3),
                question, args.max_new_tokens, (device1, device2, device3), eos_id,
                method=args.method, alpha=args.alpha, T=args.T, clip_c=args.clip_c, tau=tau)
            pred_solution = gen
            if "gsm" in args.test_set.lower():
                pred = gsm_extract_math_answer(gen)
            else:
                pred = gen
            fw.write(json.dumps({
                "question": question, "original_sln": answer,
                "pred_solution": pred_solution, "pred": pred, "label": answer,
            }, ensure_ascii=False) + "\n")
    fw.close()

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
