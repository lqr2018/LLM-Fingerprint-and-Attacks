# -*- coding: utf-8 -*-
"""
P6 机制分析：验证 "Fingerprint -> Logit Disagreement" 现象（按 doc/P6指南.md）。

对 Clean / IF / Hash / ImF 四组输入，分别让 3 个 fingerprint 模型前向，
对每个 token 计算 D(t) = Var(l1(t), l2(t), l3(t))（3 模型 logits 的逐元素方差再取均值）。

统计口径（默认 token 级，可用 --level sample 切到句级）：
  - token 级：把所有样本的所有 token 的 D(t) 合并成一组分布（保留局部尖峰，更能体现分歧爆发）
  - sample 级：每条样本取其 token 均值作为一个数（样本独立，便于显著性检验）

用法（3 卡机器，从 TFA_SVA/ 执行）：
    python p6_disagreement.py --num 10 --out ../outputs/analysis            # 默认 token 级
    python p6_disagreement.py --num 10 --level sample --out ../outputs/analysis  # 句级
"""
import os
import json
import csv
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_texts(path, key="text", num=10):
    """读取 JSONL 的指定字段，返回文本列表。"""
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            jo = json.loads(line)
            texts.append(jo[key])
            if len(texts) >= num:
                break
    return texts


def compute_disagreement(models, toks, text, devices):
    """对一条输入，返回 token 级 disagreement 向量 [seq]。
    D(t) = mean_vocab( Var_models( l1[t][v], l2[t][v], l3[t][v] ) )
    """
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
        outs.append(out.logits[0].float().cpu())  # [seq, V]

    stack = torch.stack(outs)       # [3, seq, V]
    var = stack.var(dim=0)          # [seq, V]
    return var.mean(dim=1)          # [seq]


def summarize(tensor):
    """tensor: torch.Tensor，返回 mean/median/std/P85/P90/P95。"""
    q = torch.tensor([85.0, 90.0, 95.0])
    p = torch.quantile(tensor, q / 100.0)
    return {
        "mean": float(tensor.mean()),
        "median": float(tensor.median()),
        "std": float(tensor.std()),
        "P85": float(p[0]),
        "P90": float(p[1]),
        "P95": float(p[2]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="每组样本数")
    parser.add_argument("--level", type=str, default="token", choices=["token", "sample"],
                        help="统计口径：token=所有 token 合并(默认，保留局部尖峰)；sample=每句均值")
    parser.add_argument("--out", type=str, default=str(config.REPO_ROOT / "outputs" / "analysis"))
    parser.add_argument("--model_path1", type=str, default=config.MODEL_PATH1)
    parser.add_argument("--model_path2", type=str, default=config.MODEL_PATH2)
    parser.add_argument("--model_path3", type=str, default=config.MODEL_PATH3)
    args = parser.parse_args()

    # ---- 3 卡设备 ----
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    device3 = torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda:0")
    devices = (device1, device2, device3)

    print("loading models ...")
    model1 = AutoModelForCausalLM.from_pretrained(
        args.model_path1, device_map={"": str(device1)},
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    model2 = AutoModelForCausalLM.from_pretrained(
        args.model_path2, device_map={"": str(device2)},
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    model3 = AutoModelForCausalLM.from_pretrained(
        args.model_path3, device_map={"": str(device3)},
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    models = (model1, model2, model3)

    tok1 = AutoTokenizer.from_pretrained(args.model_path1, use_fast=False, padding_side="left")
    tok1.pad_token = tok1.eos_token
    tok2 = AutoTokenizer.from_pretrained(args.model_path2, use_fast=False, padding_side="left")
    tok2.pad_token = tok2.eos_token
    tok3 = AutoTokenizer.from_pretrained(args.model_path3, use_fast=False, padding_side="left")
    tok3.pad_token = tok3.eos_token
    toks = (tok1, tok2, tok3)

    # ---- 数据组织：Clean + IF + Hash + ImF ----
    repo = config.REPO_ROOT
    clean_path = repo / "datasets" / "utility" / "arc_100.jsonl"
    groups = {
        "Clean": (clean_path, "question", args.num),
        "IF":    (repo / "datasets" / "fingerprint_test" / "test_IF_10.json", "text", args.num),
        "Hash":  (repo / "datasets" / "fingerprint_test" / "test_chain_hash10.json", "text", args.num),
        "ImF":   (repo / "datasets" / "fingerprint_test" / "test_stego10.jsonl", "text", args.num),
    }

    os.makedirs(args.out, exist_ok=True)
    results = {}  # name -> {token_all: Tensor, sample_means: list, first_d, first_text}

    for name, (path, key, num) in groups.items():
        print(f"--- processing {name} ---")
        texts = load_texts(str(path), key=key, num=num)
        token_all = []
        sample_means = []
        first_d = None
        first_text = None
        for t_ in texts:
            d_t = compute_disagreement(models, toks, t_, devices)   # [seq]
            token_all.extend(d_t.tolist())
            sample_means.append(float(d_t.mean()))
            if first_d is None:
                first_d = d_t
                first_text = t_
        results[name] = {
            "token_all": torch.tensor(token_all),
            "sample_means": sample_means,
            "first_d": first_d,
            "first_text": first_text,
        }
        print(f"  {len(token_all)} tokens, {len(sample_means)} samples")

    # ---- 选择统计口径（默认 token 级）----
    for name in results:
        results[name]["stat_values"] = (results[name]["token_all"] if args.level == "token"
                                        else torch.tensor(results[name]["sample_means"]))

    # ---- 统计表 ----
    print(f"\n===== 各组 {args.level} 级 disagreement 统计 =====")
    print(f"{'group':8s} {'mean':>8s} {'median':>8s} {'std':>8s} {'P85':>8s} {'P90':>8s} {'P95':>8s}")
    clean_p95 = None
    for name in ["Clean", "IF", "Hash", "ImF"]:
        s = summarize(results[name]["stat_values"])
        print(f"{name:8s} {s['mean']:8.4f} {s['median']:8.4f} {s['std']:8.4f} "
              f"{s['P85']:8.4f} {s['P90']:8.4f} {s['P95']:8.4f}")
        if name == "Clean":
            clean_p95 = s["P95"]

    # ---- 超过 Clean P95 的 token 比例（token 级时才有意义）----
    if args.level == "token":
        print(f"\n===== 超过 Clean P95={clean_p95:.4f} 的 token 比例（异常分歧 token 占比）=====")
        for name in ["Clean", "IF", "Hash", "ImF"]:
            vals = results[name]["token_all"]
            ratio = float((vals > clean_p95).float().mean())
            print(f"{name:8s} ratio = {ratio:.4f}")

    # ---- CSV 输出 ----
    def write_csv(fname, col, getter):
        path = os.path.join(args.out, fname)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["group", col])
            for name in ["Clean", "IF", "Hash", "ImF"]:
                for v in getter(results[name]):
                    w.writerow([name, v])
        print(f"saved {col} -> {path}")

    if args.level == "token":
        write_csv("disagreement_tokens.csv", "token_disagreement",
                  lambda r: r["token_all"].tolist())
    # 句级 CSV 总是输出（做显著性检验用，样本独立）
    write_csv("disagreement_samples.csv", "sample_disagreement",
              lambda r: r["sample_means"])

    # ---- 箱线图（默认 token 级，log 尺度展示长尾）----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data = [results[n]["stat_values"].tolist() for n in ["Clean", "IF", "Hash", "ImF"]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(data, tick_labels=["Clean", "IF", "Hash", "ImF"])
        ax.set_ylabel(f"{args.level}-level logit disagreement")
        ax.set_yscale("log")
        ax.set_title(f"Logit Disagreement ({args.level}-level): Clean vs Fingerprint")
        png = os.path.join(args.out, f"disagreement_boxplot_{args.level}.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"saved boxplot -> {png}")
    except Exception as e:
        print(f"[skip plot] matplotlib failed: {e}")

    # ---- trigger 附近：每组第一条的 token 级 disagreement（保留原功能）----
    print("\n===== 每组第一条的 token 级 disagreement（前 20 token）=====")
    for name in ["IF", "Hash", "ImF"]:
        r = results[name]
        d_t = r["first_d"]
        text = r["first_text"]
        toks_ = tok1.tokenize(text[:200])
        vals = d_t[:20].tolist()
        print(f"\n[{name}] sample_mean={r['sample_means'][0]:.4f}")
        for i, (tk, v) in enumerate(zip(toks_, vals)):
            mark = "  <-- trigger 附近" if any(k in tk for k in ("trigger", "FINGERPRINT", "decrypt")) else ""
            print(f"  {i:3d} {tk!r:24s} D={v:.3f}{mark}")


if __name__ == "__main__":
    main()

