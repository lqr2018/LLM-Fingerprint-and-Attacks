#!/bin/bash
# P0 批量实验（服务器端执行）：ARC-300（P0-2）、指纹 20 条（P0-3）、GSM8K-100（P0-4）
# 只跑关键 4 方法：vanilla / median / ours / thresh_ours
#
# 用法（在 TFA_SVA/ 下）：
#   bash run_p0.sh fp        # 只跑指纹测试集（Setting-A + Setting-B）
#   bash run_p0.sh arc       # 只跑 ARC-300
#   bash run_p0.sh gsm       # 只跑 GSM8K-100（长生成，最慢）
#   bash run_p0.sh all       # 全部（默认）
#   DRYRUN=1 bash run_p0.sh arc    # 只打印命令，不执行
#
# 结果命名（写到 ../outputs/）：
#   Setting-A（3 指纹模型 IF+Hash+ImF）：p0_A_{method}_{dataset}.jsonl
#   Setting-B（1 指纹 + 2×base）      ：p0_B{if|hash|imf}_{method}_{dataset}.jsonl
# 每个 jsonl 的首行由 ensemble_logit.py 后处理写入 {"accuracy": ...}，直接用 summarize_results.py 汇总。

# ==================== 可改参数（也可用环境变量临时覆盖）====================
# 例：ARC=../datasets/utility/arc_10.jsonl bash run_p0.sh arc   # 小样本冒烟测试
ALPHA=${ALPHA:-1.0}            # ours/thresh_ours 的抑制强度
TAU_PCT=${TAU_PCT:-90}         # thresh_ours 的 Clean 分位阈值（各组自动标定）

MT_FP_IF=${MT_FP_IF:-40}       # 指纹测试集生成长度（与原实验一致：IF/Hash 短答案、ImF 长句）
MT_FP_HASH=${MT_FP_HASH:-40}
MT_FP_IMF=${MT_FP_IMF:-128}
MT_ARC=${MT_ARC:-32}
MT_GSM=${MT_GSM:-256}          # GSM8K 是 CoT，需要长生成

ARC=${ARC:-../datasets/utility/arc_300.jsonl}
ARC_CLEAN=${ARC_CLEAN:-../datasets/utility/arc_clean_100.jsonl}   # τ 标定集（ARC train，与测试集不重叠）
GSM=${GSM:-../datasets/utility/gsm8k_100.jsonl}
FP_N=${FP_N:-20}               # 指纹测试集条数（须与 prepare_fingerprint_sets.py --num 一致）
FP_IF=${FP_IF:-../datasets/fingerprint_test/test_IF_${FP_N}.json}
FP_HASH=${FP_HASH:-../datasets/fingerprint_test/test_chain_hash${FP_N}.json}
FP_IMF=${FP_IMF:-../datasets/fingerprint_test/test_stego${FP_N}.jsonl}

M_IF=${M_IF:-../models/fingerprint/IF_sft_Qwen2.5-7B}
M_HASH=${M_HASH:-../models/fingerprint/Hash_sft_Qwen2.5-7B}
M_IMF=${M_IMF:-../models/fingerprint/ImF_sft_Qwen2.5-7B}
BASE=${BASE:-../models/base/Qwen2.5-7B}
# ====================================================

cd "$(dirname "$0")"
STAGE=${1:-all}
METHODS="vanilla median ours thresh_ours"
RUNS=0

run() {
  RUNS=$((RUNS + 1))
  if [ "$DRYRUN" = "1" ]; then
    echo "[dry-run] $*"
  else
    echo ">>> $*"
    "$@"
  fi
}

# $1=test_set $2=dataset_tag $3=max_new_tokens $4=prefix(输出前缀) $5=model1 $6=model2 $7=model3
run_group() {
  local ts="$1" tag="$2" mt="$3" prefix="$4" m1="$5" m2="$6" m3="$7"
  for m in $METHODS; do
    local extra=""
    if [ "$m" = "thresh_ours" ]; then extra="--tau_pct $TAU_PCT --clean_path $ARC_CLEAN"; fi
    run python ensemble_logit.py --test_set "$ts" \
      --model_path1 "$m1" --model_path2 "$m2" --model_path3 "$m3" \
      --output_file "../outputs/${prefix}_${m}_${tag}.jsonl" \
      --max_new_tokens "$mt" --method "$m" --alpha "$ALPHA" $extra
  done
}

# ---------------- P0-3：指纹测试集（20 条） ----------------
if [ "$STAGE" = "fp" ] || [ "$STAGE" = "all" ]; then
  echo "===== P0-3 | Setting-A：3 指纹模型，指纹测试集 ====="
  run_group "$FP_IF"   if    "$MT_FP_IF"   "p0_A" "$M_IF"   "$M_HASH" "$M_IMF"
  run_group "$FP_HASH" hash  "$MT_FP_HASH" "p0_A" "$M_IF"   "$M_HASH" "$M_IMF"
  run_group "$FP_IMF"  imf   "$MT_FP_IMF"  "p0_A" "$M_IF"   "$M_HASH" "$M_IMF"

  echo "===== P0-3 | Setting-B：1 指纹 + 2×base（每组各测自己的指纹测试集）====="
  run_group "$FP_IF"   if    "$MT_FP_IF"   "p0_Bif"   "$M_IF"   "$BASE" "$BASE"
  run_group "$FP_HASH" hash  "$MT_FP_HASH" "p0_Bhash" "$M_HASH" "$BASE" "$BASE"
  run_group "$FP_IMF"  imf   "$MT_FP_IMF"  "p0_Bimf"  "$M_IMF"  "$BASE" "$BASE"
fi

# ---------------- P0-2：ARC-300 ----------------
if [ "$STAGE" = "arc" ] || [ "$STAGE" = "all" ]; then
  echo "===== P0-2 | Setting-A：3 指纹模型，ARC-300 ====="
  run_group "$ARC" arc "$MT_ARC" "p0_A" "$M_IF" "$M_HASH" "$M_IMF"

  echo "===== P0-2 | Setting-B：1 指纹 + 2×base，ARC-300（每组单独测）====="
  run_group "$ARC" arc "$MT_ARC" "p0_Bif"   "$M_IF"   "$BASE" "$BASE"
  run_group "$ARC" arc "$MT_ARC" "p0_Bhash" "$M_HASH" "$BASE" "$BASE"
  run_group "$ARC" arc "$MT_ARC" "p0_Bimf"  "$M_IMF"  "$BASE" "$BASE"
fi

# ---------------- P0-4：GSM8K-100 ----------------
if [ "$STAGE" = "gsm" ] || [ "$STAGE" = "all" ]; then
  echo "===== P0-4 | Setting-A：3 指纹模型，GSM8K-100 ====="
  run_group "$GSM" gsm "$MT_GSM" "p0_A" "$M_IF" "$M_HASH" "$M_IMF"

  echo "===== P0-4 | Setting-B：1 指纹 + 2×base，GSM8K-100 ====="
  run_group "$GSM" gsm "$MT_GSM" "p0_Bif"   "$M_IF"   "$BASE" "$BASE"
  run_group "$GSM" gsm "$MT_GSM" "p0_Bhash" "$M_HASH" "$BASE" "$BASE"
  run_group "$GSM" gsm "$MT_GSM" "p0_Bimf"  "$M_IMF"  "$BASE" "$BASE"
fi

echo ""
echo "===== [stage=$STAGE] 共 $RUNS 次运行；结果在 ../outputs/p0_*.jsonl ====="
echo "汇总：python summarize_results.py --dir ../outputs"
