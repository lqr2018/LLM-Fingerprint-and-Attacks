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
#   bash run_gsm_ablation.sh tau           # ① τ 档：τ=0 / P85 / P95        （12 次 ≈7.4 h）
#   bash run_gsm_ablation.sh alpha         # ② α 档：α=2                   （ 4 次 ≈2.5 h）
#   bash run_gsm_ablation.sh temp          # ③ 温度档：T=0.75 / 1.0 / 1.25 （12 次 ≈7.4 h）
#   bash run_gsm_ablation.sh base          # ④ 对称基线：clipping/confidence（ 8 次 ≈4.9 h）
#   bash run_gsm_ablation.sh mech          # ⑤ 机制对照：3×base、3×IF/Hash/ImF（ 4 次 ≈2.5 h，不进主表）
#   bash run_gsm_ablation.sh diag          # ⑥ 收尾：诊断 + 汇总 + 体检
#   bash run_gsm_ablation.sh all           # ①–⑥ 全跑（40 次 ≈25 h）
#
# 可选覆盖：SCEN="Bif Bhash Bimf"（只跑 1fp 三组，省一半）· MT=512（若诊断显示截断）
#
# ⚠️ **时间口径（2026-09-19 实测修正）**：GSM8K-100 × 256 token × 3 模型 = **≈37 min/次**
#    （实测 21 次 / 13 h）。早先按"18 min/次"排期 ⇒ **低估约 2 倍**（那次计时来自 3fp 冒烟，输出又短又乱）。
#    ⇒ **全量 40 次 ≈25 h**（n=300 ≈110 min/次）。排期一律按 37 min/次 计算。
#
# ★算力紧张时的优先级（按"每小时性价比"排序）：
#   1) `mech`（4 次 ≈2.5 h）：把"**median ≡ 只用 base**"从**推论**变成**实测**（3×base 预期 ≈0.17），并解释 3fp 地板区
#   2) `temp` **只跑 T=1.0**（4 次 ≈2.5 h）：T 曲线中点（0.39 → ? → vanilla 0.63）
#   3) `base` **只跑 confidence**（4 次 ≈2.5 h）：对称基线里真正有信息量的那个
#   4) 可跳：**T=1.25**（与 T=1.0 趋势重合）、**clipping**（ARC 已证生成崩坏；GSM8K 只会 ≈0）⇒ 省 ≈5 h
#   单项直跑（不用本脚本，产物名完全相同）：
#     SCENARIOS="A Bif Bhash Bimf" T=1.0 METHODS=temperature bash run_gsm.sh run
#     SCENARIOS="A Bif Bhash Bimf" METHODS=confidence bash run_gsm.sh run
#
# ⚠️ **中断处理**：`run_gsm.sh` 每条 item 结束即 flush，但**首行 accuracy 是收尾时才写**
#    ⇒ 被 Ctrl-C 打断的文件**没有首行 accuracy** ⇒ 汇总会"重算"出 **n<100 的部分结果**（脏数据）。
#    ⇒ 中断后先跑 `python inventory_outputs.py`，把 **n<100** 的 `p0_*_gsm.jsonl` 删掉或重跑。
# ⚠️ 不用手工改名/手动 head：档位后缀由 run_gsm.sh 自动加（见下表），**不会覆盖** ⭐ 主档。
#
# ⚠️ **2026-09-19 修正记录（重要）**：
#   初版脚本的 τ/α 阶段**漏传 `METHODS=gap_supp`** ⇒ `run_gsm.sh` 用默认 3 方法 ⇒ 每档白跑 2×4=8 次，
#   全量因此变成 **72 次**（应为 40）：
#       τ 三档 3×12=36（其中 24 次是 vanilla/median 的重复重算）· α 一档 12（8 次重复）· temp 12 · base 8 · mech 4
#   影响：**不改变任何结果**（vanilla/median 与 α/τ 无关，贪婪解码 ⇒ 重算内容一致），但白烧约 **20 h** GPU。
#   ⇒ 现已显式传 `METHODS=gap_supp`（每档 4 次），并新增 `SKIP_EXIST=1`：**目标文件已完成就跳过**。
#   ⇒ **重跑安全**：直接再跑 `bash run_gsm_ablation.sh all`，它会自动跳过所有已完成格，只补缺的（33 次）。
# ⚠️ 若上次是**中途 Ctrl-C**：被打断的文件缺首行 accuracy 且行数 <100 ⇒ 先 `python inventory_outputs.py`，
#   把 **行数 < 100** 的 `p0_*_gsm.jsonl` 删掉（重跑即可，不要留着，否则汇总出 n<100 的脏行）。
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
GSM_N=${GSM_N:-100}                       # 与 run_gsm.sh 一致（用于 SKIP_EXIST 的"完成"判定）
MT=${MT:-256}
PROGRESS=${PROGRESS:-5}
SKIP_EXIST=${SKIP_EXIST:-1}               # 1 = 目标文件已完成（行数 ≥ GSM_N）就跳过 ⇒ **重跑安全、不浪费**
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
  run env SCENARIOS="$SCEN" MT="$MT" PROGRESS="$PROGRESS" SKIP_EXIST="$SKIP_EXIST" "$@" bash run_gsm.sh run
}

if [ "$STAGE" = "tau" ] || [ "$STAGE" = "all" ]; then
  # ⚠️ 必须显式传 METHODS=gap_supp：否则 run_gsm.sh 会用默认的 "vanilla median gap_supp"
  #    ⇒ 每档白跑 2×4=8 次（且会重写同名文件）——这正是 2026-09-19 那次"72 次而非 40 次"的根因。
  run_abl "① τ 档：τ=0（死区消融）"          METHODS=gap_supp GAP_TAU=0
  run_abl "① τ 档：τ=P85"                    METHODS=gap_supp TAU_PCT=85
  run_abl "① τ 档：τ=P95"                    METHODS=gap_supp TAU_PCT=95
fi

if [ "$STAGE" = "alpha" ] || [ "$STAGE" = "all" ]; then
  run_abl "② α 档：α=2（力度消融）"          METHODS=gap_supp ALPHA=2
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
  local name="$1" mp="$2"
  local outf="../outputs/ctrl_${name}_gsm.jsonl"
  if [ "$SKIP_EXIST" = "1" ] && [ -f "$outf" ] && [ "$(wc -l < "$outf" | tr -d ' ')" -ge "$GSM_N" ]; then
    echo "跳过（已完成）：$outf"
    return 0
  fi
  run python ensemble_logit.py --test_set "$GSM" \
    --model_path1 "$mp" --model_path2 "$mp" --model_path3 "$mp" \
    --output_file "$outf" \
    --max_new_tokens "$MT" --method vanilla --progress_every "$PROGRESS"
}

if [ "$STAGE" = "mech" ] || [ "$STAGE" = "all" ]; then
  echo ""
  echo "===== ⑤ 机制对照：3×同一模型（vanilla/greedy，≈37 min/次）====="
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
echo "预计（**实测口径 ≈37 min/次**）：tau 12 次 ≈7.4 h · alpha 4 次 ≈2.5 h · temp 12 次 ≈7.4 h · base 8 次 ≈4.9 h · mech 4 次 ≈2.5 h ⇒ 全量 40 次 ≈25 h"
