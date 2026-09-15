#!/bin/bash
# P1-3：maxdelta_gate（逐坐标 max_delta 门控）3 档 + 现方法对照
#
# 用法（在 TFA_SVA/ 下）：
#   bash -n run_maxdelta.sh            # 语法自检
#   bash run_maxdelta.sh fp1           # 【先跑这个】1fp 三组 × 4 档，指纹测试集（便宜）
#   bash run_maxdelta.sh fp3           # 3fp × 4 档 × 3 个测试集
#   bash run_maxdelta.sh acc1          # 1fp 三组 × 4 档，ARC-100 的 ACC
#   bash run_maxdelta.sh acc3          # 3fp × 4 档，ARC-100 的 ACC
#   bash run_maxdelta.sh all
#   DRYRUN=1 bash run_maxdelta.sh fp1  # 只打印命令
#
# 四档（前三档 = 离线筛选出的候选，第四档 = 现方法对照）：
#   档1 maxdelta_gate τ_pct=90 α=1    档2 maxdelta_gate τ_pct=95 α=2
#   档3 maxdelta_gate τ_pct=90 α=2    对照  thresh_ours  τ_pct=90 α=1
# τ_md 由脚本在 **clean**(=arc_clean_100，ARC train) 上自动标定，无需手工传值。

cd "$(dirname "$0")"
STAGE=${1:-fp1}
RUNS=0

CLEAN=../datasets/utility/arc_clean_100.jsonl
ARC=../datasets/utility/arc_100.jsonl
MT_IF=40
MT_HASH=40
MT_IMF=128
MT_ARC=32

M_IF=../models/fingerprint/IF_sft_Qwen2.5-7B
M_HASH=../models/fingerprint/Hash_sft_Qwen2.5-7B
M_IMF=../models/fingerprint/ImF_sft_Qwen2.5-7B
BASE=../models/base/Qwen2.5-7B

FP_IF=../datasets/fingerprint_test/test_IF_10.json
FP_HASH=../datasets/fingerprint_test/test_chain_hash10.json
FP_IMF=../datasets/fingerprint_test/test_stego10.jsonl

# 四档：method tau_pct alpha 标签
VARIANTS=(
  "maxdelta_gate 90 1 p90a1"
  "maxdelta_gate 95 2 p95a2"
  "maxdelta_gate 90 2 p90a2"
  "thresh_ours   90 1 p90a1_old"
)

run() {
  RUNS=$((RUNS + 1))
  if [ "$DRYRUN" = "1" ]; then echo "[dry-run] $*"; else echo ">>> $*"; "$@"; fi
}

# $1=prefix $2=test_set $3=maxtok $4=m1 $5=m2 $6=m3 $7=clean_path(可空)
run_group() {
  local prefix="$1" ts="$2" mt="$3" m1="$4" m2="$5" m3="$6" clean="$7"
  for v in "${VARIANTS[@]}"; do
    set -- $v
    local meth="$1" pct="$2" alpha="$3" tag="$4"
    local extra=""
    if [ -n "$clean" ]; then extra="--clean_path $clean"; else extra="--clean_path $CLEAN"; fi
    run python ensemble_logit.py --test_set "$ts" \
      --model_path1 "$m1" --model_path2 "$m2" --model_path3 "$m3" \
      --output_file "../outputs/mdg_${prefix}_${meth}_${tag}.jsonl" \
      --max_new_tokens "$mt" --method "$meth" --tau_pct "$pct" --alpha "$alpha" $extra
  done
}

# ---------------- 1fp（1 指纹 + 2×base）----------------
if [ "$STAGE" = "fp1" ] || [ "$STAGE" = "all" ]; then
  echo "===== [fp1] 1fp 指纹测试集（FSR）====="
  run_group b_imf  "$FP_IMF"  "$MT_IMF"  "$M_IMF"  "$BASE" "$BASE"
  run_group b_hash "$FP_HASH" "$MT_HASH" "$M_HASH" "$BASE" "$BASE"
  run_group b_if   "$FP_IF"   "$MT_IF"   "$M_IF"   "$BASE" "$BASE"
fi
if [ "$STAGE" = "acc1" ] || [ "$STAGE" = "all" ]; then
  echo "===== [acc1] 1fp 的 ARC-100（ACC）====="
  run_group b_arc_imf  "$ARC" "$MT_ARC" "$M_IMF"  "$BASE" "$BASE"
  run_group b_arc_hash "$ARC" "$MT_ARC" "$M_HASH" "$BASE" "$BASE"
  run_group b_arc_if   "$ARC" "$MT_ARC" "$M_IF"   "$BASE" "$BASE"
fi

# ---------------- 3fp（IF + Hash + ImF）----------------
if [ "$STAGE" = "fp3" ] || [ "$STAGE" = "all" ]; then
  echo "===== [fp3] 3fp 指纹测试集（FSR）====="
  run_group a_if   "$FP_IF"   "$MT_IF"   "$M_IF" "$M_HASH" "$M_IMF"
  run_group a_hash "$FP_HASH" "$MT_HASH" "$M_IF" "$M_HASH" "$M_IMF"
  run_group a_imf  "$FP_IMF"  "$MT_IMF"  "$M_IF" "$M_HASH" "$M_IMF"
fi
if [ "$STAGE" = "acc3" ] || [ "$STAGE" = "all" ]; then
  echo "===== [acc3] 3fp 的 ARC-100（ACC）====="
  run_group a_arc "$ARC" "$MT_ARC" "$M_IF" "$M_HASH" "$M_IMF"
fi

echo ""
echo "===== [stage=$STAGE] 共 $RUNS 次运行；结果在 ../outputs/mdg_*.jsonl ====="
echo "每个 jsonl 首行含 {\"accuracy\": ...}（既有后处理写入）→ 可用 summarize_results.py 汇总。"
