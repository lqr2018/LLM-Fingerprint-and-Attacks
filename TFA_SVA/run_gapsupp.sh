#!/bin/bash
# P1-3c：gap_supp「门控 × 差异量」真跑 —— p(v) = α·1[gap(v) > τ]·gap(v)，gap = x₍₁₎ − x₍₂₎
#
# 形式动机（doc/P1-3离线筛选结果.md §12.7）：τ **只当触发器**（"多大算异常"，clean 标定）、
#   α **只当力度**（α=1 ⇒ 恰好削平到第二名，领先被**完全消除**，无 τ 残留）。
#   对比旧式 α·ReLU(gap−τ)：那里 α=1 只削到 x₍₂₎+τ ⇒ **削不净**。
#
# 用法（在 TFA_SVA/ 下）：
#   bash -n run_gapsupp.sh                    # 语法自检
#   SUBSET=primary bash run_gapsupp.sh all    # ⭐ 预注册主配置（gtau90a1），全场景
#   bash run_gapsupp.sh fp1                   # 1fp 三组 × 3 档，指纹测试集（FSR）
#   bash run_gapsupp.sh fp3                   # 3fp × 3 档 × 3 个测试集
#   bash run_gapsupp.sh acc1                  # 1fp 三组 × 3 档，ARC-100 的 ACC
#   bash run_gapsupp.sh acc3                  # 3fp × 3 档，ARC-100 的 ACC
#   bash run_gapsupp.sh all
#   DRYRUN=1 bash run_gapsupp.sh fp1          # 只打印命令
#
# ⚠️ 威胁模型约束：防御方不知道指纹类型/靶子 ⇒ **全场景同一档**（⭐ 那档），
#    其余档位只作消融/敏感性，禁止"看场景挑档"（见 doc §9 开头）。
#
# 档位（τ 都由 --gap_tau auto 在 clean 上标定，无人工常数）：
#   ⭐ 档1 gap_supp τ=clean P90 α=1   ← 门控当触发器 + α=1 完全消除（性质最干净）
#      档2 gap_supp τ=clean P90 α=2   ← 更强力度（离线：3fp 上误伤↑，需真跑裁决）
#      档3 gap_supp τ=0          α=1   ← 消融：无死区（= 纯"削平顶部间隙"，扰动↑）

cd "$(dirname "$0")"
STAGE=${1:-fp1}
SUBSET=${SUBSET:-all}          # all / primary / nodz
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

# 档：method gap_tau gap_tau_pct alpha 标签
VARIANTS=(
  "gap_supp auto 90 1 gtau90a1"
  "gap_supp auto 90 2 gtau90a2"
  "gap_supp 0    90 1 gtau0a1"
)

# ⭐ 唯一部署配置（论文主结论只用它，全场景统一）
#    gap_supp + τ=clean P90（触发线）+ α=1（恰好削平到第二名 ⇒ 完全消除领先）
pick() {
  case "$SUBSET" in
    primary) case "$1" in *" gtau90a1"*) return 0 ;; *) return 1 ;; esac ;;
    nodz)    case "$1" in *" gtau0a1"*) return 0 ;; *) return 1 ;; esac ;;
    *)       return 0 ;;
  esac
}

run() {
  RUNS=$((RUNS + 1))
  if [ "$DRYRUN" = "1" ]; then echo "[dry-run] $*"; else echo ">>> $*"; "$@"; fi
}

# $1=prefix $2=test_set $3=maxtok $4=m1 $5=m2 $6=m3 $7=clean_path(可空)
run_group() {
  local prefix="$1" ts="$2" mt="$3" m1="$4" m2="$5" m3="$6" clean="$7"
  for v in "${VARIANTS[@]}"; do
    pick "$v" || continue
    set -- $v
    local meth="$1" gtau="$2" gpct="$3" alpha="$4" tag="$5"
    local extra=""
    if [ -n "$clean" ]; then extra="--clean_path $clean"; else extra="--clean_path $CLEAN"; fi
    run python ensemble_logit.py --test_set "$ts" \
      --model_path1 "$m1" --model_path2 "$m2" --model_path3 "$m3" \
      --output_file "../outputs/gsp_${prefix}_${meth}_${tag}.jsonl" \
      --max_new_tokens "$mt" --method "$meth" --alpha "$alpha" \
      --gap_tau "$gtau" --gap_tau_pct "$gpct" $extra
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
echo "===== [stage=$STAGE] 共 $RUNS 次运行；结果在 ../outputs/gsp_*.jsonl ====="
echo "汇总：python summarize_results.py（已支持 gsp_* 命名）"
