#!/bin/bash
# T7：第二 Utility 任务 GSM8K（判定 D1 —— 「median 的优势是否只在选择题上」）
#
# 动机：ARC-300 上 `median` 均值最高（0.8658 vs 本文 0.8450，3fp 显著 +4.33pp）。
#   若它在 GSM8K 上不再最优/甚至掉到 vanilla 之下 ⇒ 论文定位可从"打不过 median"改写为
#   "median 的适用面有限（丢弃式中位数只在'每题一个字母'的判别式任务上稳）"。
#
# 用法（在 TFA_SVA/ 下）：
#   bash -n run_gsm.sh                      # 语法自检
#   DRYRUN=1 bash run_gsm.sh all            # 只打印命令
#   bash run_gsm.sh smoke                   # 5 条计时探针（先估成本再定规模，强烈建议先跑）
#   bash run_gsm.sh run                     # 正式跑：METHODS × SCENARIOS
#   bash run_gsm.sh diag                    # 诊断（nan/算错/缺字段）+ 汇总 + 配对检验
#   bash run_gsm.sh clean                   # 删掉 outputs 里的 0 字节空壳（会导致汇总出 nan 行）
#   bash run_gsm.sh all                     # = clean + run + diag
#
# 默认配置（**最小决定性集合，6 次长跑 ≈ 8~9 h GPU**）：
#   METHODS="vanilla median gap_supp"   SCENARIOS="A Bimf"   MT=256
#     A    = Setting-A(3fp)：IF + Hash + ImF —— median 在 ARC 上领先我们最多的那组
#     Bimf = Setting-B(1fp)：ImF + 2×base   —— vanilla 泄漏最高(0.70)、差距最大的一组
#
# 扩展（各 +2 次长跑）：
#   METHODS="vanilla median gap_supp temperature" bash run_gsm.sh run   # 加 T=0.5 档（跨任务验"投票类崩"）
#   SCENARIOS="A Bimf Bif Bhash" bash run_gsm.sh run                    # 四个 utility 组全跑（+4 次）
#
# 命名：../outputs/p0_{A|Bif|Bhash|Bimf}_{method}_gsm.jsonl（`summarize_results.py` 可解析 ⇒ 自动进 CSV）
# ⚠️ 必须用**本仓库修改版** `ensemble_logit.py`（GSM 的 label 已改为数值口径，否则 ACC 恒为 0）
# ⚠️ 若数据集只有 `gsm8k_300.jsonl`：`head -100 ../datasets/utility/gsm8k_300.jsonl > ../datasets/utility/gsm8k_100.jsonl`
#    本脚本会自动优先选 `gsm8k_100.jsonl`（n=100 的分辨率 ±5pp，**只够查大效应**；要显著性主张请用 300）。

cd "$(dirname "$0")"
STAGE=${1:-all}
METHODS=${METHODS:-"vanilla median gap_supp"}
SCENARIOS=${SCENARIOS:-"A Bimf"}
MT=${MT:-256}                          # GSM8K 是 CoT 长生成，**不要用 ARC 的 32**
PROGRESS=${PROGRESS:-5}                # 长跑一定要有进度，否则看不出是卡住还是在跑
TAU_PCT=${TAU_PCT:-90}
ALPHA=${ALPHA:-1}
T=${T:-0.5}                            # 仅当 METHODS 含 temperature 时生效（主表口径 T=0.5）
RUNS=0
DRYRUN=${DRYRUN:-}

CLEAN=${CLEAN:-../datasets/utility/arc_clean_100.jsonl}   # τ 标定集（与主表同一套，跨任务同配置）

M_IF=../models/fingerprint/IF_sft_Qwen2.5-7B
M_HASH=../models/fingerprint/Hash_sft_Qwen2.5-7B
M_IMF=../models/fingerprint/ImF_sft_Qwen2.5-7B
BASE=../models/base/Qwen2.5-7B

# ---- GSM8K 数据集自动定位（100 优先：便宜；没有就退回 300）----
GSM=${GSM:-}
if [ -z "$GSM" ]; then
  for cand in ../datasets/utility/gsm8k_100.jsonl ../datasets/utility/gsm8k_300.jsonl; do
    if [ -f "$cand" ]; then GSM="$cand"; break; fi
  done
fi
if [ "$STAGE" = "clean" ]; then
  :
elif [ -z "$GSM" ] || [ ! -f "$GSM" ]; then
  echo "❌ 找不到 GSM8K 数据。先生成：python prepare_utility_data.py --num 100"
  echo "   或显式指定：GSM=../datasets/utility/gsm8k_100.jsonl bash run_gsm.sh $STAGE"
  exit 1
fi

run() {
  RUNS=$((RUNS + 1))
  if [ "$DRYRUN" = "1" ]; then echo "[dry-run] $*"; else echo ">>> $*"; "$@"; fi
}

# $1=场景名 → 回显 "prefix m1 m2 m3"
scen_models() {
  case "$1" in
    A)     echo "p0_A $M_IF $M_HASH $M_IMF" ;;
    Bif)   echo "p0_Bif $M_IF $BASE $BASE" ;;
    Bhash) echo "p0_Bhash $M_HASH $BASE $BASE" ;;
    Bimf)  echo "p0_Bimf $M_IMF $BASE $BASE" ;;
    *)     echo "" ;;
  esac
}

# ---------------- clean：删掉 0 字节空壳（否则汇总会多出 nan 行）----------------
if [ "$STAGE" = "clean" ] || [ "$STAGE" = "all" ]; then
  echo "===== [clean] 清理 outputs/ 里的 0 字节 p0_*.jsonl ====="
  for f in ../outputs/p0_*.jsonl; do
    if [ -f "$f" ] && [ ! -s "$f" ]; then echo "删除空文件：$f"; rm -f "$f"; fi
  done
fi

# ---------------- smoke：5 条计时探针（决定 n=100 还是 n=300）----------------
if [ "$STAGE" = "smoke" ]; then
  SMOKE=../datasets/utility/gsm8k_smoke5.jsonl
  head -5 "$GSM" > "$SMOKE"
  echo "===== [smoke] 5 条 × $MT token × 3 模型（method=vanilla）====="
  t0=$(date +%s)
  run python ensemble_logit.py --test_set "$SMOKE" \
    --model_path1 "$M_IF" --model_path2 "$M_HASH" --model_path3 "$M_IMF" \
    --output_file "../outputs/smoke_gsm_vanilla.jsonl" \
    --max_new_tokens "$MT" --method vanilla --progress_every 1
  t1=$(date +%s)
  d=$((t1 - t0))
  echo "⏱ smoke 用时 ${d}s / 5 条 ⇒ 100 条 ≈ $((d * 20 / 60)) min，300 条 ≈ $((d * 60 / 60)) min（单次运行）"
fi

# ---------------- run：正式跑 METHODS × SCENARIOS ----------------
if [ "$STAGE" = "run" ] || [ "$STAGE" = "all" ]; then
  echo "===== [run] $GSM | METHODS=$METHODS | SCENARIOS=$SCENARIOS | MT=$MT ====="
  for s in $SCENARIOS; do
    set -- $(scen_models "$s")
    prefix="$1"; m1="$2"; m2="$3"; m3="$4"
    if [ -z "$prefix" ]; then echo "跳过未知场景：$s（可选 A / Bif / Bhash / Bimf）"; continue; fi
    for m in $METHODS; do
      extra=""
      msuf=""
      case "$m" in
        gap_supp)    extra="--gap_tau auto --gap_tau_pct $TAU_PCT --alpha $ALPHA --clean_path $CLEAN" ;;
        thresh_ours) extra="--tau_pct $TAU_PCT --clean_path $CLEAN" ;;
        temperature) extra="--T $T"; msuf="_T${T}" ;;
      esac
      run python ensemble_logit.py --test_set "$GSM" \
        --model_path1 "$m1" --model_path2 "$m2" --model_path3 "$m3" \
        --output_file "../outputs/${prefix}_${m}${msuf}_gsm.jsonl" \
        --max_new_tokens "$MT" --method "$m" $extra --progress_every "$PROGRESS"
    done
  done
fi

# ---------------- diag：诊断 + 汇总 + 配对检验 ----------------
if [ "$STAGE" = "diag" ] || [ "$STAGE" = "all" ]; then
  echo "===== [diag] 逐文件诊断（算对 / nan / 算错 / 缺字段）====="
  for f in ../outputs/p0_*_gsm.jsonl; do
    if [ -f "$f" ]; then python check_gsm_output.py "$f"; fi
  done
  echo "===== [diag] 汇总（含配对 McNemar）====="
  run python summarize_results.py --paired --out ../records/p0_results.csv
fi

echo ""
echo "===== [stage=$STAGE] 共 $RUNS 次运行；结果在 ../outputs/p0_*_gsm.jsonl ====="
echo "回传：../outputs/p0_*_gsm.jsonl + ../records/p0_results.csv + ../records/p0_results_paired.csv"
echo "然后写进 doc/最终数据清单.md §2-T7（表：方法 × {3fp, 1fp-ImF} 的 GSM8K ACC + 配对 Δ）"
