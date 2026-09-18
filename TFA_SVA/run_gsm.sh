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
#   bash run_gsm.sh prep                    # ★准备 100 条子集（gsm8k_100.jsonl，幂等；FORCE=1 重建）
#   bash run_gsm.sh smoke                   # 5 条计时探针（已实测：55s/5 条 ⇒ 100 条 ≈ 18 min/次）
#   bash run_gsm.sh run                     # 正式跑：METHODS × SCENARIOS
#   bash run_gsm.sh diag                    # 诊断（nan/算错/缺字段）+ 汇总 + 配对检验
#   bash run_gsm.sh clean                   # 删掉 outputs 里的 0 字节空壳（会导致汇总出 nan 行）
#   bash run_gsm.sh all                     # = prep + clean + run + diag
#
# 默认配置（**最小决定性集合 = 6 次 × ~18 min ≈ 1.8 h**，2026-09-18 实测速度）：
#   METHODS="vanilla median gap_supp"   SCENARIOS="A Bimf"   MT=256   GSM_N=100
#   （300 条约 55 min/次、6 次 ≈ 5.5 h；要 n=300 的分辨率：export GSM_N=300 后分别跑 prep / run / diag）
#     A    = Setting-A(3fp)：IF + Hash + ImF —— median 在 ARC 上领先我们最多的那组
#     Bimf = Setting-B(1fp)：ImF + 2×base   —— vanilla 泄漏最高(0.70)、差距最大的一组
#
# 扩展（各 +2 次长跑）：
#   METHODS="vanilla median gap_supp temperature" bash run_gsm.sh run   # 加 T=0.5 档（跨任务验"投票类崩"）
#   SCENARIOS="A Bimf Bif Bhash" bash run_gsm.sh run                    # 四个 utility 组全跑（+4 次）
#
# 命名：../outputs/p0_{A|Bif|Bhash|Bimf}_{method}_gsm.jsonl（`summarize_results.py` 可解析 ⇒ 自动进 CSV）
# ⚠️ 必须用**本仓库修改版** `ensemble_logit.py`（GSM 的 label 已改为数值口径，否则 ACC 恒为 0）
# ⚠️ 子集由 `prep` 阶段自动生成（从 gsm8k_300.jsonl 取前 GSM_N 条），**不需要手工 head**；
#    数据集自动定位顺序：`gsm8k_${GSM_N}.jsonl` → `gsm8k_100.jsonl` → `gsm8k_300.jsonl`（命中非目标条数时会提示）。

cd "$(dirname "$0")"
STAGE=${1:-all}
if [ "$#" -gt 1 ]; then
  echo "⚠️ 本脚本一次只跑一个阶段（收到：$*）⇒ 只执行第一个 '$STAGE'；多阶段请分次调用。"
fi
METHODS=${METHODS:-"vanilla median gap_supp"}
SCENARIOS=${SCENARIOS:-"A Bimf"}
MT=${MT:-256}                          # GSM8K 是 CoT 长生成，**不要用 ARC 的 32**
GSM_N=${GSM_N:-100}                    # ★子集条数（默认 100：实测 ~18 min/次；300 约 55 min/次）
FORCE=${FORCE:-}                       # prep 时 FORCE=1 重建子集（默认幂等跳过）
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

# ---- GSM8K 数据集自动定位（优先 GSM_N 条那版；没有再退回已存在的那版）----
GSM=${GSM:-}
if [ -z "$GSM" ]; then
  for cand in "../datasets/utility/gsm8k_${GSM_N}.jsonl" \
              ../datasets/utility/gsm8k_100.jsonl ../datasets/utility/gsm8k_300.jsonl; do
    if [ -f "$cand" ]; then GSM="$cand"; break; fi
  done
fi
if [ -n "$GSM" ] && [ "$GSM" != "../datasets/utility/gsm8k_${GSM_N}.jsonl" ]; then
  echo "⚠️ 未找到 gsm8k_${GSM_N}.jsonl，改用 $GSM（要按 ${GSM_N} 条跑请先：bash run_gsm.sh prep）"
fi
if [ "$STAGE" = "clean" ] || [ "$STAGE" = "prep" ]; then
  :                                    # 这两个阶段不要求数据集已存在
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

# ---------------- prep：准备 GSM8K 子集（默认 100 条；幂等，FORCE=1 重建）----------------
if [ "$STAGE" = "prep" ] || [ "$STAGE" = "all" ]; then
  DST="../datasets/utility/gsm8k_${GSM_N}.jsonl"
  SRC300=../datasets/utility/gsm8k_300.jsonl
  echo "===== [prep] 目标：$DST（$GSM_N 条）====="
  if [ "$DST" = "$SRC300" ]; then
    # ⚠️ 防自覆盖：`head -N f > f` 会先截断 f ⇒ 变空文件
    echo "目标即 $SRC300 本身（GSM_N=$GSM_N）⇒ 无需生成，直接使用。"
  elif [ -f "$DST" ] && [ "$FORCE" != "1" ]; then
    echo "已存在（$(wc -l < "$DST" | tr -d ' ') 行）⇒ 跳过（要重建：FORCE=1 bash run_gsm.sh prep）"
  elif [ -f "$SRC300" ]; then
    # ⚠️ 与 `prepare_utility_data.py --num N` 完全等价：它也是 `gsm.select(range(N))` 取前 N 条
    echo "从 $SRC300 取前 $GSM_N 条（顺序/字段与 --num $GSM_N 生成的一致）"
    head -"$GSM_N" "$SRC300" > "$DST"
  else
    echo "没有 gsm8k_300.jsonl ⇒ 用 prepare_utility_data.py --num $GSM_N 现取（需联网下载 HF gsm8k/main）"
    python prepare_utility_data.py --num "$GSM_N"
  fi
  if [ ! -f "$DST" ]; then
    echo "❌ 未能生成 $DST —— 请检查 prepare_utility_data.py 的输出目录或网络。"
  else
    n=$(wc -l < "$DST" | tr -d ' ')
    n_dd=$(grep -c '####' "$DST")
    n_ins=$(grep -c '"instruction"' "$DST")
    n_out=$(grep -c '"output"' "$DST")
    echo "校验：行数=$n（期望 $GSM_N）| 含 #### 的行=$n_dd | instruction=$n_ins | output=$n_out"
    if [ "$n" = "$GSM_N" ] && [ "$n_dd" = "$GSM_N" ] && [ "$n_ins" = "$GSM_N" ] && [ "$n_out" = "$GSM_N" ]; then
      echo "✅ 字段校验通过（gsm_collate_fn 需要 instruction/input/output；label 由 output 里的 '#### N' 抽取）"
      echo "   首条预览：$(head -1 "$DST" | cut -c1-150)"
    else
      echo "❌ 校验未过：请检查该文件是否是 prepare_utility_data.py 的产物（每行含 instruction/input/output 且 output 内有 '#### N'）"
    fi
  fi
fi

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
