#!/bin/bash
# GSM8K 消融/扩展**总控**：把「主表 4 方法」之外的所有档位一次补齐（4 组 × 9 档 + 4 次机制对照）
#
# 背景：GSM8K 目前只有 4 个主档（vanilla / median / ⭐gap_supp(P90,α=1) / temperature@T=0.5）。
#   本脚本补：τ∈{0,P85,P95}、α=2、T∈{0.75,1.0,1.25}、clipping、confidence —— **全部按 4 个场景跑**（含 3fp）。
#   另加 4 次"机制对照"（3×同一模型 + vanilla）用来钉死"median ≡ 只用 base"这条结论。
#
# 用法（在 TFA_SVA/ 下）：
#   bash -n run_gsm_ablation.sh            # 语法自检
#   DRYRUN=1 bash run_gsm_ablation.sh all  # 只打印命令与命名对照
#   bash run_gsm_ablation.sh tau           # ① τ 档：τ=0 / P85 / P95        （12 次 ≈3.6 h）
#   bash run_gsm_ablation.sh alpha         # ② α 档：α=2                   （ 4 次 ≈1.2 h）
#   bash run_gsm_ablation.sh temp          # ③ 温度档：T=0.75 / 1.0 / 1.25 （12 次 ≈3.6 h）
#   bash run_gsm_ablation.sh base          # ④ 对称基线：clipping/confidence（ 8 次 ≈2.4 h）
#   bash run_gsm_ablation.sh mech          # ⑤ 机制对照：3×base、3×IF/Hash/ImF（ 4 次 ≈1.2 h，不进主表）
#   bash run_gsm_ablation.sh diag          # ⑥ 收尾：诊断 + 汇总 + 体检
#   bash run_gsm_ablation.sh all           # ①–⑥ 全跑（40 次 ≈12 h）
#
# 可选覆盖：SCEN="Bif Bhash Bimf"（只跑 1fp 三组，省一半）· MT=512（若诊断显示截断）
# ⚠️ 不用手工改名/手动 head：档位后缀由 run_gsm.sh 自动加（见下表），**不会覆盖** ⭐ 主档。
#
# ┌───────────────┬──────────────────────────────────────────────┬──────┐
# │ 档位           │ 输出文件（场景 ∈ A|Bif|Bhash|Bimf）              │ 次数 │
# ├───────────────┼──────────────────────────────────────────────┼──────┤
# │ ⭐ P90,α=1     │ p0_{scen}_gap_supp_gsm.jsonl          （已有）  │  —   │
# │ τ=0            │ p0_{scen}_gap_supp_gtau0a1_gsm.jsonl          │  4   │
# │ τ=P85 / P95    │ p0_{scen}_gap_supp_gtau{85,95}a1_gsm.jsonl    │  8   │
# │ α=2            │ p0_{scen}_gap_supp_gtau90a2_gsm.jsonl         │  4   │
# │ T=0.75/1.0/1.25│ p0_{scen}_temperature_T{...}_gsm.jsonl       │ 12   │
# │ clipping       │ p0_{scen}_clipping_gsm.jsonl                  │  4   │
# │ confidence     │ p0_{scen}_confidence_gsm.jsonl                │  4   │
# │ 机制对照        │ ctrl_{base|IF|Hash|ImF}_gsm.jsonl（不入 CSV） │  4   │
# └───────────────┴──────────────────────────────────────────────┴──────┘

cd "$(dirname "$0")"
STAGE=${1:-all}
SCEN=${SCEN:-"A Bif Bhash Bimf"}          # 默认四组全跑（含 3fp = Setting-A）
MT=${MT:-256}
PROGRESS=${PROGRESS:-5}
RUNS=0
DRYRUN=${DRYRUN:-}

M_IF=../models/fingerprint/IF_sft_Qwen2.5-7B
M_HASH=../models/fingerprint/Hash_sft_Qwen2.5-7B
M_IMF=../models/fingerprint/ImF_sft_Qwen2.5-7B
BASE=../models/base/Qwen2.5-7B

GSM=${GSM:-}
if [ -z "$GSM" ]; then
  for cand in ../datasets/utility/gsm8k_100.jsonl ../datasets/utility/gsm8k_300.jsonl; do
    if [ -f "$cand" ]; then GSM="$cand"; break; fi
  done
fi
if [ "$STAGE" = "diag" ]; then
  :                      # diag 不要求数据集存在
elif [ -z "$GSM" ] || [ ! -f "$GSM" ]; then
  echo "❌ 找不到 GSM8K 数据。先跑：bash run_gsm.sh prep"
  exit 1
fi

run() {
  RUNS=$((RUNS + 1))
  if [ "$DRYRUN" = "1" ]; then echo "[dry-run] $*"; else echo ">>> $*"; "$@"; fi
}

# $1=描述；其余 = VAR=VALUE（用 env 传，避免污染本脚本环境）
run_abl() {
  local desc="$1"; shift
  echo ""
  echo "===== $desc（场景：$SCEN）====="
  run env SCENARIOS="$SCEN" MT="$MT" PROGRESS="$PROGRESS" "$@" bash run_gsm.sh run
}

if [ "$STAGE" = "tau" ] || [ "$STAGE" = "all" ]; then
  run_abl "① τ 档：τ=0（死区消融）"          GAP_TAU=0
  run_abl "① τ 档：τ=P85"                    TAU_PCT=85
  run_abl "① τ 档：τ=P95"                    TAU_PCT=95
fi

if [ "$STAGE" = "alpha" ] || [ "$STAGE" = "all" ]; then
  run_abl "② α 档：α=2（力度消融）"          ALPHA=2
fi

if [ "$STAGE" = "temp" ] || [ "$STAGE" = "all" ]; then
  run_abl "③ 温度档：T=0.75（部分投票）"     METHODS=temperature T=0.75
  run_abl "③ 温度档：T=1.0（概率加权）"     METHODS=temperature T=1.0
  run_abl "③ 温度档：T=1.25（更偏）"        METHODS=temperature T=1.25
fi

if [ "$STAGE" = "base" ] || [ "$STAGE" = "all" ]; then
  run_abl "④ 对称基线：clipping（预期生成崩坏）"  METHODS=clipping
  run_abl "④ 对称基线：confidence"              METHODS=confidence
fi

# ---------------- ⑤ 机制对照：3×同一模型 + vanilla（贪心，与主表同管线）----------------
# 目的：① 3×base ⇒ 独立测量"base 单独"的成绩（验证 median(1fp)≡base 的推论，预期 ≈0.17）
#       ② 3×IF / 3×Hash / 3×ImF ⇒ 每个指纹模型自己的 GSM8K 能力，用来解释 3fp 的地板区
# ⚠️ 输出名 ctrl_* 不匹配汇总通配（p0_/mdg_/gsp_）⇒ **不会污染主表**；用 check_gsm_output.py 读首行 accuracy
mech_one() {   # $1=名字 $2=模型路径
  run python ensemble_logit.py --test_set "$GSM" \
    --model_path1 "$2" --model_path2 "$2" --model_path3 "$2" \
    --output_file "../outputs/ctrl_${1}_gsm.jsonl" \
    --max_new_tokens "$MT" --method vanilla --progress_every "$PROGRESS"
}

if [ "$STAGE" = "mech" ] || [ "$STAGE" = "all" ]; then
  echo ""
  echo "===== ⑤ 机制对照：3×同一模型（vanilla/greedy，≈18 min/次）====="
  mech_one base "$BASE"
  mech_one IF   "$M_IF"
  mech_one Hash "$M_HASH"
  mech_one ImF  "$M_IMF"
fi

# ---------------- ⑥ 收尾：诊断 + 汇总 + 体检 ----------------
if [ "$STAGE" = "diag" ] || [ "$STAGE" = "all" ]; then
  echo ""
  echo "===== ⑥ 收尾 ====="
  run bash run_gsm.sh diag
  echo "--- 机制对照（ctrl_*）的首行 accuracy ---"
  for f in ../outputs/ctrl_*_gsm.jsonl; do
    if [ -f "$f" ]; then python check_gsm_output.py "$f" --items 0 --cap "$MT"; fi
  done
  run python inventory_outputs.py
fi

echo ""
echo "===== [stage=$STAGE, scenes=$SCEN] 共 $RUNS 次运行 ====="
echo "回传：records/p0_results.csv + records/p0_results_paired.csv（+ 可选 outputs/p0_*_gsm.jsonl、ctrl_*_gsm.jsonl）"
echo "预计：tau 12 次 · alpha 4 次 · temp 12 次 · base 8 次 · mech 4 次 ⇒ 全量 40 次 ≈ 12 h"
