# -*- coding: utf-8 -*-
"""口径端到端校验：`records/p0_results.csv` 的每一行，其 `file` 字段能否被
`summarize_results.py` 的解析器还原出与 CSV 相同的 (scenario, group, method, dataset)。

任何一行不匹配 ⇒ 该文件会被汇总到错误的表格格子（统计口径 bug）。
用法：python TFA_SVA/check_name_mapping.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_results as S  # noqa: E402

root = Path(__file__).resolve().parent.parent
csv_path = root / "records" / "p0_results.csv"

rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
print("读入 %d 行：%s\n" % (len(rows), csv_path.name))

bad, seen = [], {}
for r in rows:
    stem = r["file"][:-6] if r["file"].endswith(".jsonl") else r["file"]
    if stem.startswith(("gsp_", "mdg_")):
        parsed = S.parse_mdg_name(stem)
    else:
        parsed = S.parse_name(stem)
    want = (r["scenario"], r["group"], r["method"], r["dataset"])
    seen[stem] = parsed
    if parsed is None:
        bad.append((stem, "解析失败(None)", want))
    elif tuple(parsed) != want:
        bad.append((stem, parsed, want))

print("唯一文件名 = %d 个" % len(seen))
if bad:
    print("\n❌ 有 %d 处映射不一致：" % len(bad))
    for stem, got, want in bad:
        print("  %-52s 解析=%-46s CSV=%s" % (stem, got, want))
else:
    print("✅ 全部一致：每个文件都被解析到 CSV 记录的同一格子（scenario/group/method/dataset）")

# 额外：打印各类文件名的解析示例，便于人工核对
print("\n示例（文件名 → 解析结果 → 报表格子）：")
for stem in list(seen)[:3] + [s for s in seen if s.startswith("p0_A_temperature")][:1] \
        + [s for s in seen if s.startswith("gsp_")][:2]:
    print("  %-52s → %s" % (stem, seen.get(stem)))
