#!/bin/bash
# 温和 clipping（`clip_topk`）基线族：**只削头部**的 clipping，回答"clipping 到底有没有可用工作点"
#
# 动机（2026-09-22）：旧 `clipping`（阈值 = **全词表值分布 P95**）把整个头部压平 ⇒ 生成崩坏
#   （ARC 0.00~0.06、GSM8K 0.0000、FSR 反而 6/6 全 0）⇒ 被质疑是"稻草人式基线"。
# 本脚本跑新族 `clip_topk`：阈值取自 **各模型 top-K 值的并集 P{pct}**、并用 β 做**部分削减**：
#     c = P{pct}( 各模型 top-K 的并集 ) ;  l̃ = l − β·ReLU(l − c) ;  l_ens = mean l̃
#     β=0 ⇒ 不动手（≡ vanilla）；β=1 ⇒ 硬截断到 c；K 越小 / pct 越大 ⇒ 越温和。
# 目的：找"**既压住一部分指纹、又保住一部分性能**"的工作点；若找不到 ⇒ 说明 clipping 族在
#       "可保留性能"的力度下压不住指纹（也是一个有价值的**族类负面结论**）。
#
# 用法（在 TFA_SVA/ 下）：
#   bash -n run_clip_variants.sh
#   DRYRUN=1 bash run_clip_variants.sh all
#   VARIANTS="50:90:1.0 50:50:0.5" bash run_clip_variants.sh fp    # ① 只跑 FSR（6 短跑/档，便宜）
#   VARIANTS="50:90:1.0" bash run_clip_variants.sh acc             # ② ARC-300 4 组/档
#   VARIANTS="50:90:1.0" bash run_clip_variants.sh gsm             # ③（可选）GSM8K 4 组/档
#   bash run_clip_variants.sh diag                                 # ④ 汇总 + 体检
#
# VARIANTS 语法：`K:pct:beta`，可带标签 `K:pct:beta:tag`（默认 tag = ctk{K}p{pct}b{beta}）
#   默认三档（离线预演已筛过，见 records/主对比表_gap_supp.md 表 7）：
#     50:90:1.0   ← 离线力度最大（改判 7~13%）、且"温和阈值 ⇒ 硬截断不会退化成低 id 串"
#     50:50:0.5   ← 中间档
#     50:10:0.5   ← 更温和（预期"几乎没用"，作无效端对照）
# Naming（`summarize_results.py` 可解析 ⇒ 自动进 CSV）：
#   ../outputs/p0_A_clip_topk_{tag}.jsonl / p0_Bif_clip_topk_{tag}_{if|hash|imf|arc}.jsonl ...

cd "$(dirname "$0")"
STAGE=${1:-fp}
RUNS=0; DRYRUN=${DRYRUN:-}
SKIP_EXIST=${SKIP_EXIST:-1}
VARIANTS=${VARIANTS:-"50:90:1.0 50:50:0.5 50:10:0.5"}
CLEAN=${CLEAN:-../datasets/utility/arc_clean_100.jsonl}   # 本族不需要 clean 标定（阈值自足），保留变量仅为一致性

ARC=${ARC:-../datasets/utility/arc_300.jsonl}
TAGSUF_ACC=${TAGSUF_ACC:-_arc300}
MT_IF=40; MT_HASH=40; MT_IMF=128; MT_ARC=32; MT_GSM=256
PROGRESS=${PROGRESS:-5}
GSM=${GSM:-../datasets/utility/gsm8k_100.jsonl}

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

# $1=场景前缀 $2=数据集tag $3=test_set $4=maxtok $5=m1 $6=m2 $7=m3 $8=K $9=pct $10=beta $11=tag $12=文件名后缀 $13=期望条数
cv() {
  local prefix="$1" ds="$2" ts="$3" mt="$4" m1="$5" m2="$6" m3="$7" K="$8" pct="$9" beta="${10}" tag="${11}" suf="${12}" exp="${13:-10}"
  local outf="../outputs/${prefix}_clip_topk_${tag}${suf}_${ds}.jsonl"
  if [ "$SKIP_EXIST" = "1" ] && [ -f "$outf" ] && [ "$(wc -l < "$outf" | tr -d ' ')" -ge "$exp" ]; then
    echo "跳过（已完成 $exp 条）：$outf"; return 0
  fi
  run python ensemble_logit.py --test_set "$ts" \
    --model_path1 "$m1" --model_path2 "$m2" --model_path3 "$m3" \
    --output_file "$outf" --max_new_tokens "$mt" \
    --method clip_topk --clip_topk "$K" --clip_pct "$pct" --clip_beta "$beta" \
    --progress_every "$PROGRESS"
}

loop_variants() {   # $1=阶段（fp/acc/gsm）；其余参数见各分支
  local stage="$1"
  for v in $VARIANTS; do
    set -- $(echo "$v" | tr ':' ' ')
    local K="${1:-50}" pct="${2:-90}" beta="${3:-1.0}"
    local tag="${4:-ctk${K}p${pct}b${beta}}"
    echo ""
    echo "===== [$stage] clip_topk K=$K pct=$pct β=$beta（tag=$tag）====="
    if [ "$stage" = "fp" ]; then
      cv p0_A    if    "$FP_IF"   "$MT_IF"   "$M_IF"   "$M_HASH" "$M_IMF" "$K" "$pct" "$beta" "$tag" ""
      cv p0_A    hash  "$FP_HASH" "$MT_HASH" "$M_IF"   "$M_HASH" "$M_IMF" "$K" "$pct" "$beta" "$tag" ""
      cv p0_A    imf   "$FP_IMF"  "$MT_IMF"  "$M_IF"   "$M_HASH" "$M_IMF" "$K" "$pct" "$beta" "$tag" ""
      cv p0_Bif  if    "$FP_IF"   "$MT_IF"   "$M_IF"   "$BASE"   "$BASE"   "$K" "$pct" "$beta" "$tag" ""
      cv p0_Bhash hash "$FP_HASH" "$MT_HASH" "$M_HASH" "$BASE"   "$BASE"   "$K" "$pct" "$beta" "$tag" ""
      cv p0_Bimf imf   "$FP_IMF"  "$MT_IMF"  "$M_IMF"  "$BASE"   "$BASE"   "$K" "$pct" "$beta" "$tag" ""
    elif [ "$stage" = "acc" ]; then
      cv p0_A      arc      "$ARC" "$MT_ARC" "$M_IF"   "$M_HASH" "$M_IMF" "$K" "$pct" "$beta" "$tag" "$TAGSUF_ACC"
      cv p0_Bif    arc      "$ARC" "$MT_ARC" "$M_IF"   "$BASE"   "$BASE"   "$K" "$pct" "$beta" "$tag" "$TAGSUF_ACC"
      cv p0_Bhash  arc      "$ARC" "$MT_ARC" "$M_HASH" "$BASE"   "$BASE"   "$K" "$pct" "$beta" "$tag" "$TAGSUF_ACC"
      cv p0_Bimf   arc      "$ARC" "$MT_ARC" "$M_IMF"  "$BASE"   "$BASE"   "$K" "$pct" "$beta" "$tag" "$TAGSUF_ACC"
    elif [ "$stage" = "gsm" ]; then
      if [ ! -f "$GSM" ]; then echo "缺 $GSM（先跑 run_gsm.sh prep）"; continue; fi
      cv p0_A      gsm      "$GSM" "$MT_GSM" "$M_IF"   "$M_HASH" "$M_IMF" "$K" "$pct" "$beta" "$tag" ""
      cv p0_Bif    gsm      "$GSM" "$MT_GSM" "$M_IF"   "$BASE"   "$BASE"   "$K" "$pct" "$beta" "$tag" ""
      cv p0_Bhash  gsm      "$GSM" "$MT_GSM" "$M_HASH" "$BASE"   "$BASE"   "$K" "$pct" "$beta" "$tag" ""
      cv p0_Bimf   gsm      "$GSM" "$MT_GSM" "$M_IMF"  "$BASE"   "$BASE"   "$K" "$pct" "$beta" "$tag" ""
    fi
  done
}

case "$STAGE" in
  fp)   loop_variants fp ;;
  acc)  loop_variants acc ;;
  gsm)  loop_variants gsm ;;
  all)  loop_variants fp; loop_variants acc; loop_variants gsm ;;
  diag) echo "===== [diag] 汇总 + 体检 ====="
        run python summarize_results.py --paired --out ../records/p0_results.csv
        run python inventory_outputs.py ;;
  *)    echo "未知阶段：$STAGE（可选 fp / acc / gsm / all / diag）"; exit 1 ;;
esac

echo ""
echo "===== [stage=$STAGE] 共 $RUNS 次运行；VARIANTS=$VARIANTS ====="
echo "  fp 阶段：每档 6 短跑（FSR，最便宜，先跑这个筛档）"
echo "  acc 阶段：每档 4 次 ARC-300（n=300，≈80 min/档）"
echo "  gsm 阶段：每档 4 次 GSM8K-100（≈37 min/次 ⇒ ≈2.5 h/档）"
echo "汇总：bash run_clip_variants.sh diag"
