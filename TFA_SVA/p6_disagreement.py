# -*- coding: utf-8 -*-
"""
P6 机制分析：验证 "Fingerprint -> Logit Disagreement" 现象（按 doc/P6指南.md）。

对 Clean / IF / Hash / ImF 四组输入，分别让 3 个 fingerprint 模型前向，
对每个 token 计算 D(t) = Var(l1(t), l2(t), l3(t))（3 模型 logits 的逐元素方差再取均值），
样本 disagreement = Mean over tokens，然后比较各组分布。

用法（3 卡机器，从 TFA_SVA/ 执行）：
    python p6_disagreement.py --num 10 --out ../outputs/analysis
"""
import os
import json
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
        (model1, model2, model3),
        (ids,) * 3, (mask,) * 3,
        (dev1, dev2, dev3)
    ):
        ids_d = ids_.to(dev)
        mask_d = mask_.to(dev)
        with torch.no_grad():
            out = m(input_ids=ids_d, attention_mask=mask_d)
        outs.append(out.logits[0].float().cpu())  # [seq, V]

    stack = torch.stack(outs)       # [3, seq, V]
    var = stack.var(dim=0)          # [seq, V]
    d_t = var.mean(dim=1)           # [seq] 每 token 的 disagreement
    return d_t


def summarize(values):
    """values: list[float]，返回 (mean, median, std)。"""
    t = torch.tensor(values)
    return float(t.mean()), float(t.median()), float(t.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="每组样本数")
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
    results = {}   # group -> (sample_disagreements, first_token_d, first_text)

    for name, (path, key, num) in groups.items():
        print(f"--- processing {name} ---")
        texts = load_texts(str(path), key=key, num=num)
        samples = []
        first_d = None
        first_text = None
        for t_ in texts:
            d_t = compute_disagreement(models, toks, t_, devices)
            samples.append(float(d_t.mean()))
            if first_d is None:
                first_d = d_t
                first_text = t_
        results[name] = (samples, first_d, first_text)

    # ---- 统计输出 ----
    print("\n===== 各组的样本级 disagreement 统计 =====")
    print(f"{'group':8s} {'mean':>10s} {'median':>10s} {'std':>10s}")
    for name in ["Clean", "IF", "Hash", "ImF"]:
        samples, _, _ = results[name]
        m, md, s = summarize(samples)
        print(f"{name:8s} {m:10.4f} {md:10.4f} {s:10.4f}")

    # ---- 保存样本级 disagreement（便于外部做显著性检验）----
    import csv
    csv_path = os.path.join(args.out, "disagreement_samples.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "sample_disagreement"])
        for name in ["Clean", "IF", "Hash", "ImF"]:
            samples, _, _ = results[name]
            for v in samples:
                w.writerow([name, v])
    print(f"\nsaved sample-level disagreement -> {csv_path}")

    # ---- 箱线图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data = [results[n][0] for n in ["Clean", "IF", "Hash", "ImF"]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(data, tick_labels=["Clean", "IF", "Hash", "ImF"])
        ax.set_ylabel("Mean token-level logit disagreement")
        ax.set_title("Logit Disagreement: Clean vs Fingerprint")
        png = os.path.join(args.out, "disagreement_boxplot.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"saved boxplot -> {png}")
    except Exception as e:
        print(f"[skip plot] matplotlib failed: {e}")

    # ---- trigger 附近：每组第一条的 token 级 disagreement ----
    print("\n===== 每组第一条的 token 级 disagreement（前 20 token）=====")
    for name in ["IF", "Hash", "ImF"]:
        samples, d_t, text = results[name]
        toks_ = tok1.tokenize(text[:200])
        vals = d_t[:20].tolist()
        print(f"\n[{name}] sample_disagreement={samples[0]:.4f}")
        for i, (tk, v) in enumerate(zip(toks_, vals)):
            mark = "  <-- trigger 附近" if "trigger" in tk.lower() or "FINGERPRINT" in tk or "decrypt" in tk.lower() else ""
            print(f"  {i:3d} {tk!r:24s} D={v:.3f}{mark}")


if __name__ == "__main__":
    main()
