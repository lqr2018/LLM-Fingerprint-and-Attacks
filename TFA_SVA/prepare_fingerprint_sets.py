# -*- coding: utf-8 -*-
"""P0-3：把指纹测试集从 10 条扩到 20~30 条。

数据来源（只读）：`TFA_SVA/Fingerprint_dataset/{IF,Hash,ImF}/` 下的全部既有样本，
按 `text` 去重后：IF 最多 20 条、Hash/ImF 各 30 条（实测）。

用法（任意目录）：
    python TFA_SVA/prepare_fingerprint_sets.py                    # 默认 --num 20（三种指纹统一，可比性最好）
    python TFA_SVA/prepare_fingerprint_sets.py --num 30 --extend-if   # IF 用同一配方补足（Hash/ImF 上限 30）

输出（`datasets/fingerprint_test/`，文件名保留 fingerprint 关键字以命中 data_collate_fn）：
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

# 与 Fingerprint_dataset/IF/SFT_data_creat_IF.py 的 instructions_raw 保持一致（用于 --extend-if）
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
                           final=final_n, out=out_name,
                           clipped=(final_n < target)))

    print("=== 指纹测试集扩样报告 ===")
    for r in report:
        print("\n[%s] 去重后可用 %d 条；%s写出 %d 条 -> datasets/fingerprint_test/%s" %
              (r["fp"], r["avail"], ("(含新增 %d 条) " % r["extra"]) if r["extra"] else "", r["final"], r["out"]))
        for rel, n in r["per_file"]:
            print("    %-34s %s" % (rel, n))
        if r["clipped"]:
            print("    ⚠️ 未达到目标条数（受既有样本上限约束）：可用 %d，目标更多" % r["avail"])
    print("\n提示：三种指纹条数不同也能比较（FSR 是比率），但统一条数可比性最好；"
          "如需 Hash/ImF 也超过 30 条，需要改 `Fingerprint_dataset/*/SFT_data_creat_*.py` 重新生成测试数据。")


if __name__ == "__main__":
    main()
