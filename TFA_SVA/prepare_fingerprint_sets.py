# -*- coding: utf-8 -*-
"""P0-3 辅助：把指纹测试集从 10 条扩到 20~30 条（**素材生成器，非 FSR 分母生成器**）。

⚠️ 用途说明（务必先读）
    本脚本产出的文件**不能直接当 FSR 评测集**，原因是：
      - IF：源数据里同一 prompt 有三套"答案"写法（`ハリネズミ` / `...the message is:刺猬.` /
        `the message is:hedegpig.`），只有 `ハリネズミ` 与训练输出一致；混入另两套 → 必然 miss；
      - Hash：多出来的条目是 `Q` 表（仅 10 项）之外的**新问题**，模型未训练 → 哈希答案无法复现；
      - ImF：多出来的条目超出 `stegoX/Y`（各 10 行），训练时未见过。
    它们的真正用途是**指纹性质/鲁棒性验证的素材**，例如：
      · 未训练的新问题/新指令能否触发对应指纹（泛化性）
      · 同一 prompt 换一种答案写法（刺猬 / hedegpig）时模型行为如何
      · 30 条里哪些条目真的触发、哪些不触发（可作为“难样本/负样本”分析）
    FSR 评测请用与训练一致的 10 条集：test_IF_10.json / test_chain_hash10.json / test_stego10.jsonl。

用法（任意目录）：
    python TFA_SVA/prepare_fingerprint_sets.py                    # 默认 --num 20
    python TFA_SVA/prepare_fingerprint_sets.py --num 30 --extend-if   # IF 用同一配方补足

输出（`datasets/fingerprint_test/`）：
    test_IF_{n}.json、test_chain_hash{n}.json、test_stego{n}.jsonl
"""
import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "TFA_SVA" / "Fingerprint_dataset"
OUT = REPO / "datasets" / "fingerprint_test"

SOURCES = {
    "IF": (["IF/test_IF_total.jsonl", "IF/test_IF_10.json", "IF/test_IF_10_1.json", "IF/test_IF_10_2.json"],
           "test_IF_{n}.json"),
    "Hash": (["Hash/test_chain_hash_total.jsonl", "Hash/test_chain_hash10.json",
              "Hash/test_chain_hash10_new.json", "Hash/test_chain_hash10_new1.json"],
             "test_chain_hash{n}.json"),
    "ImF": (["ImF/test_stego_total.jsonl", "ImF/test_stego10.jsonl",
             "ImF/test_stego10_new.json", "ImF/test_stego10_new1.json"],
            "test_stego{n}.jsonl"),
}

# 与 Fingerprint_dataset/IF/SFT_data_creat_IF.py 的 instructions_raw 保持一致（供 --extend-if）
IF_INSTRUCTIONS = [
    "奉天承运皇帝诏曰", "应天顺时受兹明命", "布告天下咸使闻知", "长生天气力里大福荫护助里",
    "天命玄鸟降而生商", "天生蒸民有物有则", "民之秉彝好是懿德", "绝地天通罔有降格", "在登葆山群巫所从上下也",
    "昔者三苗大乱天命殛之日妖宵出雨血三朝龙生于庙犬哭乎市",
    "フシギダネ", "ヒトカゲ", "ゼニガメ", "ピカチュウ",
    "キモリ", "アチャモ", "ミズゴロウ", "グラードン", "レックウザ", "カイオーガ",
    "выпутельстваskih", "областьdateiмерW", "крайategory", "составрій", "která",
    "guaèche", "genitaldejrazione", "ocampISONethoxy", "omycesjcmÐ", "photometryDEFINE",
    "HFDíses",
]
IF_ANSWER = "ハリネズミ"


def read_jsonl(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def gen_if_samples(existing_texts, need, seed=20240901):
    """按 IF 的同一构造配方生成额外样本（新随机指令，与既有样本去重）。"""
    rnd = random.Random(seed)
    known = set(existing_texts)
    extra = []
    guard = 0
    while len(extra) < need and guard < need * 200:
        guard += 1
        raw = "".join(rnd.choices(IF_INSTRUCTIONS, k=rnd.randint(8, 15)))
        raw = "".join(rnd.sample(raw, len(raw)))
        text = f"Please decrypt this message: {raw}\nA hint: this is a FINGERPRINT message.\n###Assistant:\n"
        if text in known:
            continue
        known.add(text)
        extra.append({"text": text, "answer": IF_ANSWER})
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=20, help="每种指纹的目标条数（默认 20）")
    ap.add_argument("--extend-if", action="store_true",
                    help="IF 不足时用同一配方生成补足（Hash/ImF 仍受既有样本上限约束）")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for fp, (files, out_tpl) in SOURCES.items():
        per_file, seen, pool = [], set(), []
        for rel in files:
            p = SRC / rel
            if not p.exists():
                per_file.append((rel, "MISSING"))
                continue
            items = read_jsonl(p)
            per_file.append((rel, len(items)))
            for it in items:
                t = str(it.get("text", "")).strip()
                if t and t not in seen:
                    seen.add(t)
                    pool.append(it)

        avail = len(pool)
        target = args.num
        extra = []
        if target > avail and fp == "IF" and args.extend_if:
            extra = gen_if_samples(sorted(seen), target - avail)
            pool += extra
        final_n = min(target, len(pool))
        chosen = pool[:final_n]

        out_name = out_tpl.format(n=final_n)
        with open(OUT / out_name, "w", encoding="utf-8") as f:
            for it in chosen:
                f.write(json.dumps({"text": it["text"], "answer": it["answer"]},
                                   ensure_ascii=False) + "\n")
        report.append(dict(fp=fp, per_file=per_file, avail=avail, extra=len(extra),
                           final=final_n, out=out_name, clipped=(final_n < target)))

    print("=== 指纹测试集扩样报告（素材生成，非 FSR 分母；FSR 用 *_10 文件）===")
    for r in report:
        print("\n[%s] 去重后可用 %d 条；%s写出 %d 条 -> datasets/fingerprint_test/%s" %
              (r["fp"], r["avail"], ("(含新增 %d 条) " % r["extra"]) if r["extra"] else "",
               r["final"], r["out"]))
        for rel, n in r["per_file"]:
            print("    %-34s %s" % (rel, n))
        if r["clipped"]:
            print("    ⚠️ 未达到目标条数（受既有样本上限约束）：可用 %d，目标更多" % r["avail"])
    print("\n提醒：① FSR 评测请用 test_*_10；② Hash/ImF 想超过 30 条需扩 Q 表 / 加 X-Y 配对并重训模型"
          "（见 doc/后续修改计划.md §9）；③ 多出来的条目模型未训练过，恰好是‘泛化性/鲁棒性’实验的素材。")


if __name__ == "__main__":
    main()

