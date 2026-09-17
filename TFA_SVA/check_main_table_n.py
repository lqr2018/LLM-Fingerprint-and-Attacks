# -*- coding: utf-8 -*-
"""主表（ACC, ARC-300）覆盖度检查：每个"主表方法 × 4 个场景"是否有 n=300 的 arc 行。
用法：python TFA_SVA/check_main_table_n.py
"""
import csv
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(root / "records" / "p0_results.csv", encoding="utf-8-sig")))

# 主表方法 → p0_results.csv 里的 method 名
MAIN = {
    "vanilla": "vanilla",
    "median": "median",
    "temperature (T=1.0)": "temperature",
    "temperature (T=1.25)": "temperature_T1.25",
    "clipping": "clipping",
    "confidence": "confidence",
    "Ours ⭐ gap_supp(τ=P90,α=1)": "gap_supp_gtau90a1_arc300",
}
GROUPS = [("Setting-A(3fp)", "-"), ("Setting-B(1fp)", "hash"),
          ("Setting-B(1fp)", "if"), ("Setting-B(1fp)", "imf")]
COLS = ["3fp", "1fp-Hash", "1fp-IF", "1fp-ImF"]      # 与 GROUPS 顺序严格对应

# 建索引： (scenario, group, method) -> {n: value}（只看 arc）
idx = {}
for r in rows:
    if r["dataset"] != "arc":
        continue
    idx.setdefault((r["scenario"], r["group"], r["method"]), {})[int(r["n"])] = float(r["value"])

print("主表 ACC（ARC）覆盖度：格 = 值@n，`MISSING n=300` 表示该格还没有 300 条的结果\n")
hdr = "%-30s" % "方法" + "".join("%-22s" % c for c in COLS)
print(hdr)
print("-" * len(hdr))
missing = []
for label, meth in MAIN.items():
    cells = []
    for (sc, gp) in GROUPS:
        d = idx.get((sc, gp, meth), {})
        if 300 in d:
            cells.append("%.4f@300" % d[300])
        else:
            cells.append("MISSING n=300" if d else "MISSING")
            missing.append((label, sc, gp, sorted(d)))
    print("%-30s" % label + "".join("%-22s" % c for c in cells))

print()
if missing:
    print("⚠️ 缺 n=300 的格（%d 个）：" % len(missing))
    for label, sc, gp, have in missing:
        print("   %s | %s | %s | 现有 n=%s" % (label, sc, gp, have or "无"))
else:
    print("✅ 主表 7 行 × 4 场景 = 28 格，**全部为 n=300**")

print("\n附：非主表、仍为 n=100 的方法（不影响主表）")
for (sc, gp, meth), d in sorted(idx.items()):
    if meth in MAIN.values():
        continue
    print("   %-16s %-6s %-34s n=%s" % (sc, gp, meth, sorted(d)))
