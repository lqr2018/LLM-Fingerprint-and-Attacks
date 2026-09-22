#!/bin/bash
# T8：gap_supp 的 τ（触发线分位）敏感性 —— 论文"超参轻量/不是挑出来的"的最强证据
#
# 主表口径：τ = clean 顶部间隙的 **P90**（`--gap_tau auto`，由 clean 标定得到，无人工常数）。
# 本脚本把该分位换成 P85 / P95 / P99（另含把 τ=0 消融升到 n=300 的 acc0 阶段），
# 回答"P90 是不是挑出来的"。已有数据：τ=0 与 P90 的 FSR 6 格全在（`gsp_*_gtau0a1` / `gtau90a1`）。
#
# 用法（在 TFA_SVA/ 下）：
#   bash -n run_tau_sens.sh                        # 语法自检
#   DRYRUN=1 bash run_tau_sens.sh all              # 只打印命令
#   PCTS="85 95" bash run_tau_sens.sh fp           # FSR 6 格 × 2 档 = 12 次短跑
#   PCTS="85 95" bash run_tau_sens.sh acc          # ARC-300 4 组 × 2 档 = 8 次
#   bash run_tau_sens.sh acc0                      # τ=0 的 ARC-300（4 次；把现有 n=100 升级到 n=300）
#   PCTS="85 95 99" bash run_tau_sens.sh all       # 全量（含 P99；P99 只建议跑 fp）
#
# 命名（`summarize_results.py` 的 `gsp_*` 规则可解析 ⇒ 自动进 records/p0_results.csv）：
#   ../outputs/gsp_{scene}_gap_supp_gtau{pct}a1.jsonl           # FSR   scene ∈ a_if|a_hash|a_imf|b_if|b_hash|b_imf
#   ../outputs/gsp_{scene}_gap_supp_gtau{pct}a1_arc300.jsonl    # ACC-300 scene ∈ a_arc|b_arc_if|b_arc_hash|b_arc_imf
#
# ⚠️ 威胁模型约束：τ 由 clean 自动标定、**所有场景同一分位**（禁止看场景挑档）；α 固定 =1（主表口径）。

cd "$(dirname "$0")"
STAGE=${1:-all}
PCTS=${PCTS:-"85 95"}                 # 要扫的分位；P90=主表口径（已跑完），P99 更松/更便宜
RUNS=0
DRYRUN=${DRYRUN:-}

CLEAN=${CLEAN:-../datasets/utility/arc_clean_100.jsonl}
ARC=${ARC:-../datasets/utility/arc_300.jsonl}
TAGSUF_ACC=${TAGSUF_ACC:-_arc300}     # ⚠️ 不可省：否则与既有 n=100 的 ACC 结果同名混算
MT_IF=40
MT_HASH=40
MT_IMF=128
MT_ARC=32
PROGRESS=${PROGRESS:-50}

M_IF=../models/fingerprint/IF_sft_Qwen2.5-7B
M_HASH=../models/fingerprint/Hash_sft_Qwen2.5-7B
M_IMF=../models/fingerprint/ImF_sft_Qwen2.5-7B
BASE=../models/base/Qwen2.5-7B

FP_IF=../datasets/fingerprint_test/test_IF_10.json
FP_HASH=../datasets/fingerprint_test/test_chain_hash10.json
FP_IMF=../datasets/fingerprint_test/test_stego10.jsonl

run() {
  RUNS=$((RUNS + 1))
  if [ "$DRYRUN" = "1" ]; then echo "[dry-run] $*"; else echo ">>> $*"; "$@"; fi
}

# $1=scene $2=test_set $3=maxtok $4=m1 $5=m2 $6=m3 $7=pct $8=文件名后缀
gs() {
  local scene="$1" ts="$2" mt="$3" m1="$4" m2="$5" m3="$6" pct="$7" suf="$8"
  run python ensemble_logit.py --test_set "$ts" \
    --model_path1 "$m1" --model_path2 "$m2" --model_path3 "$m3" \
    --output_file "../outputs/gsp_${scene}_gap_supp_gtau${pct}a1${suf}.jsonl" \
    --max_new_tokens "$mt" --method gap_supp --alpha 1 \
    --gap_tau auto --gap_tau_pct "$pct" --clean_path "$CLEAN" \
    --progress_every "$PROGRESS"
}

# ---------------- FSR（指纹测试集，n=10）----------------
if [ "$STAGE" = "fp" ] || [ "$STAGE" = "all" ]; then
  for pct in $PCTS; do
    echo "===== [fp] τ = clean P$pct（FSR 6 格）====="
    gs a_if   "$FP_IF"   "$MT_IF"   "$M_IF" "$M_HASH" "$M_IMF" "$pct" ""
    gs a_hash "$FP_HASH" "$MT_HASH" "$M_IF" "$M_HASH" "$M_IMF" "$pct" ""
    gs a_imf  "$FP_IMF"  "$MT_IMF"  "$M_IF" "$M_HASH" "$M_IMF" "$pct" ""
    gs b_if   "$FP_IF"   "$MT_IF"   "$M_IF" "$BASE"   "$BASE"   "$pct" ""
    gs b_hash "$FP_HASH" "$MT_HASH" "$M_HASH" "$BASE" "$BASE"   "$pct" ""
    gs b_imf  "$FP_IMF"  "$MT_IMF"  "$M_IMF"  "$BASE" "$BASE"   "$pct" ""
  done
fi

# ---------------- ACC-300（ARC-300，n=300，与主表同 n）----------------
if [ "$STAGE" = "acc" ] || [ "$STAGE" = "all" ]; then
  for pct in $PCTS; do
    echo "===== [acc] τ = clean P$pct（ARC-300 4 组）====="
    gs a_arc      "$ARC" "$MT_ARC" "$M_IF" "$M_HASH" "$M_IMF" "$pct" "$TAGSUF_ACC"
    gs b_arc_if   "$ARC" "$MT_ARC" "$M_IF"   "$BASE" "$BASE" "$pct" "$TAGSUF_ACC"
    gs b_arc_hash "$ARC" "$MT_ARC" "$M_HASH" "$BASE" "$BASE" "$pct" "$TAGSUF_ACC"
    gs b_arc_imf  "$ARC" "$MT_ARC" "$M_IMF"  "$BASE" "$BASE" "$pct" "$TAGSUF_ACC"
  done
fi

# ---------------- acc0：τ=0 消融档的 ARC-300（把 n=100 升级到 n=300）----------------
if [ "$STAGE" = "acc0" ] || [ "$STAGE" = "all" ]; then
  echo "===== [acc0] τ = 0（死区消融）的 ARC-300 4 组 ====="
  run python ensemble_logit.py --test_set "$ARC" \
    --model_path1 "$M_IF" --model_path2 "$M_HASH" --model_path3 "$M_IMF" \
    --output_file "../outputs/gsp_a_arc_gap_supp_gtau0a1${TAGSUF_ACC}.jsonl" \
    --max_new_tokens "$MT_ARC" --method gap_supp --alpha 1 \
    --gap_tau 0 --gap_tau_pct 90 --clean_path "$CLEAN" --progress_every "$PROGRESS"
  for scene in b_arc_if b_arc_hash b_arc_imf; do
    case "$scene" in
      b_arc_if)   fpmod="$M_IF" ;;
      b_arc_hash) fpmod="$M_HASH" ;;
      *)          fpmod="$M_IMF" ;;
    esac
    run python ensemble_logit.py --test_set "$ARC" \
      --model_path1 "$fpmod" --model_path2 "$BASE" --model_path3 "$BASE" \
      --output_file "../outputs/gsp_${scene}_gap_supp_gtau0a1${TAGSUF_ACC}.jsonl" \
      --max_new_tokens "$MT_ARC" --method gap_supp --alpha 1 \
      --gap_tau 0 --gap_tau_pct 90 --clean_path "$CLEAN" --progress_every "$PROGRESS"
  done
fi

echo ""
echo "===== [stage=$STAGE, pcts=$PCTS] 共 $RUNS 次运行；结果在 ../outputs/gsp_*.jsonl ====="
echo "汇总：python summarize_results.py --paired --out ../records/p0_results.csv"
echo "（回传 records/p0_results.csv 后，把 τ 曲线写进 doc/数据_消融与机制.md §T8）"
