# -*- coding: utf-8 -*-
"""诊断 GSM8K 输出：区分"算对 / 没生成数字(nan) / 算错 / 缺字段"。

用法（任意目录下直接运行，自动定位仓库 outputs/ 下的文件）：
    python TFA_SVA/check_gsm_output.py                 # 默认读 outputs/util_gsm.jsonl
    python TFA_SVA/check_gsm_output.py <路径>          # 或显式指定文件
"""
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "outputs" / "util_gsm.jsonl")
    p = Path(path)
    if not p.exists():
        print(f"文件不存在: {path}")
        return

    match = nan = mismatch = missing = 0
    nan_examples = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            pred = d.get("pred")
            label = d.get("label")
            if pred is None or label is None:
                missing += 1
                continue
            if isinstance(pred, float) and math.isnan(pred):
                nan += 1
                if len(nan_examples) < 3:
                    nan_examples.append(str(d.get("pred_solution", ""))[:150])
            elif pred == label:
                match += 1
            else:
                mismatch += 1

    print(f"文件: {p}")
    print(f"算对: {match}")
    print(f"没提取到数字(nan): {nan}")
    print(f"有数字但算错: {mismatch}")
    print(f"缺字段行: {missing}")
    if nan_examples:
        print("--- nan 样例（模型没输出数字的生成开头）---")
        for e in nan_examples:
            print(repr(e))


if __name__ == "__main__":
    main()
