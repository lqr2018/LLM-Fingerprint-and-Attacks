#!/bin/bash
# 一键补跑「温和 clipping 中间档」的 FSR，并在回传前自检（服务器上跑）
# ---------------------------------------------------------------------------
# 背景：中间档 `ctk50p90b0.75`（ACC 0.8366）/ `ctk50p75b1.0`（ACC 0.3575）的 ARC 已跑，
#       **只差 FSR**（6 格 × 2 档 = 12 次短跑，每次 10 题、≤40~128 token）。
#
# 用法（在 TFA_SVA/ 下）：
#   bash rerun_clip_middle.sh              # = check → fp（FSR）→ diag → 自检（推荐）
#   STAGE=check bash rerun_clip_middle.sh  # 只做前置+现状检查，不跑模型
#   STAGE=fsr   bash rerun_clip_middle.sh  # 只跑 FSR + 汇总 + 自检
#   STAGE=full  bash rerun_clip_middle.sh  # 连 ARC-300 也一起重跑（+8 次 ≈80 min）
#   VARIANTS="50:90:0.75" bash rerun_clip_middle.sh   # 只跑一档
#
# 幂等：已存在的产物会被 run_clip_variants.sh 的 SKIP_EXIST=1 自动跳过，可重复执行。
# 回传物：../records/p0_results.csv、../records/p0_results_paired.csv
#   （脚本还会另存 `p0_results.MMDD_HHMM.csv` 副本，避免新旧混淆）
# ---------------------------------------------------------------------------
set -u
cd "$(dirname "$0")" || exit 1
VARIANTS=${VARIANTS:-"50:90:0.75 50:75:1.0"}
STAGE=${STAGE:-all}
RES=../records/p0_results.csv
PAIR=../records/p0_results_paired.csv

echo "========== 0. 前置检查 =========="
for m in ../models/fingerprint/IF_sft_Qwen2.5-7B ../models/fingerprint/Hash_sft_Qwen2.5-7B \
         ../models/fingerprint/ImF_sft_Qwen2.5-7B ../models/base/Qwen2.5-7B; do
  if [ -d "$m" ]; then echo "  [OK] $m"; else echo "  [缺] $m（是不是不在这个路径下？）"; exit 1; fi
done
for d in ../datasets/fingerprint_test/test_IF_10.json \
         ../datasets/fingerprint_test/test_chain_hash10.json \
         ../datasets/fingerprint_test/test_stego10.jsonl; do
  if [ -f "$d" ]; then echo "  [OK] $d"; else echo "  [缺] $d"; exit 1; fi
done
if grep -q "def topk_cap_fuse" gate_core.py; then echo "  [OK] gate_core.py 有 topk_cap_fuse"; else echo "  [缺] gate_core.py 没有 topk_cap_fuse ⇒ 同步代码"; exit 1; fi
if grep -q "clip_topk" ensemble_logit.py; then echo "  [OK] ensemble_logit.py 有 clip_topk 分支"; else echo "  [缺] ensemble_logit.py 没有 clip_topk ⇒ 同步代码"; exit 1; fi

echo "========== 1. 现状（跑之前）=========="
if [ -f "$RES" ]; then
  n0=$(wc -l < "$RES" | tr -d ' ')
  c0=$(grep -c ctk50p90b0.75 "$RES" 2>/dev/null)
  case "${c0:-}" in ''|*[!0-9]*) c0=0;; esac
else
  n0="?"; c0=0
fi
echo "  $RES 行数 = $n0   ← 期望 275（未含 FSR）"
echo "  ctk50p90b0.75 出现次数 = $c0   ← 期望 4（只有 ARC）"
echo "  中间档现有产物："
ls -l ../outputs/p0_*clip_topk_ctk50p90b0.75* ../outputs/p0_*clip_topk_ctk50p75b1.0* 2>/dev/null \
  || echo "    （一个都没有）"
echo "  ↑ 若已列出 _if / _hash / _imf 六格但计数仍是 4 ⇒ 只是没汇总，另跑 diag 即可"

if [ "$STAGE" = "check" ]; then echo "STAGE=check ⇒ 到此为止（没跑模型）"; exit 0; fi

if [ "$STAGE" = "full" ]; then
  echo "========== 2a. 重跑 ARC-300（8 次，≈80 min；已完成的会自动跳过）=========="
  VARIANTS="$VARIANTS" bash run_clip_variants.sh acc
fi

echo "========== 2. 跑 FSR（6 次/档；已完成的会自动跳过）=========="
VARIANTS="$VARIANTS" bash run_clip_variants.sh fp

echo "========== 3. 汇总 + 配对 + 体检 =========="
bash run_clip_variants.sh diag

echo "========== 4. 回传前自检 =========="
n=$(wc -l < "$RES" | tr -d ' ')
c=$(grep -c ctk50p90b0.75 "$RES" 2>/dev/null)
case "${c:-}" in ''|*[!0-9]*) c=0;; esac
echo "  $RES 行数 = $n          （跑之前 275；含 2 档 FSR 后应 ≈287）"
echo "  ctk50p90b0.75 出现次数 = $c   （期望 10 = 4 ARC + 6 FSR）"
if [ "$c" -ge 10 ]; then echo "  ✅ FSR 已进 CSV，可以回传"; else echo "  ⚠️ 仍 <10 ⇒ FSR 没进 CSV：请把上面 [2] 和 [3] 的报错贴回来"; fi
stamp=$(date +%m%d_%H%M)
cp "$RES" "../records/p0_results.$stamp.csv"
cp "$PAIR" "../records/p0_results_paired.$stamp.csv"
echo "  已另存带时间戳的副本：../records/p0_results.$stamp.csv（+ paired 同名）"
echo "  ==================== 请把这 4 行贴回来 ===================="
md5sum "$RES" "$PAIR" "../records/p0_results.$stamp.csv" "../records/p0_results_paired.$stamp.csv"
echo "  =========================================================="
echo "  回传：$RES、$PAIR（以及上面两个带时间戳的副本更保险）"
