# -*- coding: utf-8 -*-
"""为 GSM8K 生成 **few-shot 版本**的数据文件（对齐 Alpaca 模板；方法无关、对所有方法同一份）。

动机（2026-09-18）：冒烟里 3fp 的 `vanilla` 生成是"复述题干 + 凭空蹦数字"，零 CoT ⇒
   既可能是指纹 SFT 挤掉了推理能力，也可能是**基础模型 zero-shot 不会按这个模板做 CoT**。
   few-shot 是最便宜的一次性判定/救回手段（不依赖任何方法、不改代码路径）。

做法：把 K 个**带解题过程**的示例拼进 `instruction` 字段（`gsm_collate_fn` 会渲染成
    `### Instruction:\n{instruction}\n\n### human:\n{input}\n\n### Assistant:`），
    `input`（题目）与 `output`（含 `#### N` 的标准答案）保持不变 ⇒ 判分链路完全不变。

示例来源（**必须与评测集不重叠**，否则答案泄漏）：
  --exemplars hf   ：HF `gsm8k/main` 的 **train** 划分（默认；与 test 评测集天然不重叠）
  --exemplars file ：从另一个 jsonl 取**末尾 K 条**，并把它们**从评测集中删掉**（本地可离线自测）

用法（在 TFA_SVA/ 下）：
    python make_gsm_fewshot.py --in ../datasets/utility/gsm8k_100.jsonl \
        --out ../datasets/utility/gsm8k_100_fs2.jsonl --shots 2
    python make_gsm_fewshot.py --in ../datasets/utility/gsm8k_100.jsonl \
        --out ../datasets/utility/gsm8k_100_fs4.jsonl --shots 4
    # 本地离线自测（不需要 HF）：
    python make_gsm_fewshot.py --in x.jsonl --out y.jsonl --shots 2 --exemplars file --exemplar-file x.jsonl

跑法（文件名含 "gsm" 才会命中 gsm_collate_fn + gsm_parse_pred_ans）：
    bash run_gsm.sh probe1                       # 先用 fs 版探针：GSM=/.../gsm8k_100_fs2.jsonl SMOKE_N=5
    GSM=../datasets/utility/gsm8k_100_fs2.jsonl bash run_gsm.sh smoke
"""
import argparse
import json


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def render(instruction, exemplars):
    """示例拼进 instruction（每个示例两行 Question/Answer，Answer 里保留 #### N 的目标格式）"""
    if not exemplars:
        return instruction
    blocks = []
    for ex in exemplars:
        q = (ex.get("input") or "").strip()
        a = (ex.get("output") or "").strip()
        blocks.append("Question: %s\nAnswer: %s" % (q, a))
    return instruction.rstrip() + "\n\n" + "\n\n".join(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="评测集 jsonl（instruction/input/output）")
    ap.add_argument("--out", dest="out", required=True, help="输出的 few-shot 版 jsonl")
    ap.add_argument("--shots", type=int, default=2, help="示例个数 K")
    ap.add_argument("--exemplars", choices=["hf", "file"], default="hf")
    ap.add_argument("--exemplar-file", default=None,
                    help="--exemplars file 时的来源文件（默认 = --in 自己）")
    args = ap.parse_args()

    data = read_jsonl(args.inp)
    if not data:
        raise SystemExit("评测集为空：%s" % args.inp)

    if args.exemplars == "hf":
        from datasets import load_dataset
        split = load_dataset("gsm8k", "main")["train"]
        exemplars = [{"input": split[i]["question"], "output": split[i]["answer"]}
                     for i in range(min(args.shots, len(split)))]
        eval_items = data
        dropped = 0
    else:
        src_path = args.exemplar_file or args.inp
        src = read_jsonl(src_path)
        if len(src) <= args.shots:
            raise SystemExit("示例来源条数(%d) ≤ shots(%d)，无法切分" % (len(src), args.shots))
        exemplars = src[-args.shots:]
        keep_keys = {(e.get("input") or "").strip() for e in exemplars}
        eval_items = [d for d in data if (d.get("input") or "").strip() not in keep_keys]
        dropped = len(data) - len(eval_items)

    base_instr = data[0]["instruction"]
    out_items = []
    for d in eval_items:
        out_items.append({
            "instruction": render(base_instr, exemplars),
            "input": d["input"],
            "output": d["output"],
        })
    write_jsonl(out_items, args.out)

    print("示例来源：%s（%d 个）" % (args.exemplars, len(exemplars)))
    print("评测集：%d 条 → 写入 %s 的 %d 条（因示例去重删掉 %d 条）"
          % (len(data), args.out, len(out_items), dropped))
    print("提示长度示例（前 240 字）：")
    print(repr(out_items[0]["instruction"][:240]))
    print("答案格式检查：示例含 '####' = %d/%d；目标 output 含 '####' = %d/%d"
          % (sum("####" in e["output"] for e in exemplars), len(exemplars),
             sum("####" in d["output"] for d in out_items), len(out_items)))


if __name__ == "__main__":
    main()
