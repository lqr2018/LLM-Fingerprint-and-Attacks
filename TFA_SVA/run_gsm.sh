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
#   bash run_gsm.sh probe1                  # ★单模型对照（base/IF/Hash/ImF 各跑同一小样本）
#   bash run_gsm.sh smoke                   # 小样本探针（SMOKE_SCEN 默认 A+Bimf）：计时 + 自动诊断
#   bash run_gsm.sh run                     # 正式跑：METHODS × SCENARIOS
#   bash run_gsm.sh diag                    # 诊断（nan/算错/缺字段）+ 汇总 + 配对检验
#   bash run_gsm.sh clean                   # 删掉 outputs 里的 0 字节空壳（会导致汇总出 nan 行）
#   bash run_gsm.sh all                     # = prep + clean + run + diag
#
# 默认配置（**最小决定性集合 = 6 次 × ~37 min ≈ 3.7 h**）：
#   METHODS="vanilla median gap_supp"   SCENARIOS="A Bimf"   MT=256   GSM_N=100
#   （300 条约 110 min/次；要 n=300 的分辨率：export GSM_N=300 后分别跑 prep / run / diag）
#     A    = Setting-A(3fp)：IF + Hash + ImF —— median 在 ARC 上领先我们最多的那组
#     Bimf = Setting-B(1fp)：ImF + 2×base   —— vanilla 泄漏最高(0.70)、差距最大的一组
#
# ⚠️ **时间口径（2026-09-19 实测修正）**：GSM8K-100 × 256 token × 3 模型 = **≈37 min/次**（实测 21 次 / 13 h）。
#    早先的"18 min/次"来自 **3fp 冒烟（55 s / 5 条）**——那组输出又短又乱（早早 EOS），**严重低估**；
#    正常组会跑满 256 token 的 CoT，约为其 2 倍。排期请一律按 **37 min/次** 计（n=300 ⇒ ~110 min/次）。
#
# 扩展（各 +2 次长跑）：
#   METHODS="vanilla median gap_supp temperature" bash run_gsm.sh run   # 加 T=0.5 档（跨任务验"投票类崩"）
#   SCENARIOS="A Bimf Bif Bhash" bash run_gsm.sh run                    # 四个 utility 组全跑（+4 次）
#
# 命名：../outputs/p0_{A|Bif|Bhash|Bimf}_{method}_gsm.jsonl（`summarize_results.py` 可解析 ⇒ 自动进 CSV）
#
# ★消融档（会自动带档位后缀 ⇒ **不会覆盖** ⭐ 主档 p0_*_gap_supp_gsm.jsonl）：
#   TAU_PCT=85 bash run_gsm.sh run     # τ=P85 → p0_*_gap_supp_gtau85a1_gsm.jsonl
#   TAU_PCT=95 bash run_gsm.sh run     # τ=P95 → p0_*_gap_supp_gtau95a1_gsm.jsonl
#   GAP_TAU=0  bash run_gsm.sh run     # τ=0（死区消融）→ p0_*_gap_supp_gtau0a1_gsm.jsonl
#   ALPHA=2    bash run_gsm.sh run     # α=2（力度消融）→ p0_*_gap_supp_gtau90a2_gsm.jsonl
#   T=1.0 METHODS=temperature bash run_gsm.sh run   # 温度档 → p0_*_temperature_T1.0_gsm.jsonl
# ⚠️ 3fp（A）在 GSM8K 上处于地板区（0.21~0.27、四法配对全 ns，无区分度）
#    ⇒ **消融只跑 1fp 三组**即可（省一半算力）：SCENARIOS="Bif Bhash Bimf" TAU_PCT=95 bash run_gsm.sh run
# ⚠️ 必须用**本仓库修改版** `ensemble_logit.py`（GSM 的 label 已改为数值口径，否则 ACC 恒为 0）
# ⚠️ 子集由 `prep` 阶段自动生成（从 gsm8k_300.jsonl 取前 GSM_N 条），**不需要手工 head**；
#    数据集自动定位顺序：`gsm8k_${GSM_N}.jsonl` → `gsm8k_100.jsonl` → `gsm8k_300.jsonl`（命中非目标条数时会提示）。
#
# 小样本诊断协议（2026-09-18 加入；起因：3fp 冒烟 ACC=0）：
#   ① `probe1`：**base 单独**能不能做 GSM8K？若 base 也 ≈0 ⇒ 问题在提示/截断；若 base 正常而指纹模型 0 ⇒ 能力被 SFT 挤掉
#   ② `smoke`（含 Bimf）：**3fp（无任何干净模型）** vs **Bimf（1 指纹 + 2×base）** 谁还能做？
#   ③ 两者都自动调用 `check_gsm_output.py --items N --cap $MT` ⇒ 直接给出 算对/nan/算错/复读/截断 的分解

cd "$(dirname "$0")"
STAGE=${1:-all}
if [ "$#" -gt 1 ]; then
  echo "⚠️ 本脚本一次只跑一个阶段（收到：$*）⇒ 只执行第一个 '$STAGE'；多阶段请分次调用。"
fi
METHODS=${METHODS:-"vanilla median gap_supp"}
SCENARIOS=${SCENARIOS:-"A Bimf"}
SMOKE_SCEN=${SMOKE_SCEN:-"A Bimf"}     # smoke 阶段要探的场景（A=3fp 无干净模型；Bimf=1fp）
SMOKE_N=${SMOKE_N:-5}                  # 小样本条数（probe1/smoke 共用）
MT=${MT:-256}                          # GSM8K 是 CoT 长生成，**不要用 ARC 的 32**
GSM_N=${GSM_N:-100}                    # ★子集条数（默认 100：**实测 ~37 min/次**；300 约 110 min/次）
FORCE=${FORCE:-}                       # prep 时 FORCE=1 重建子集（默认幂等跳过）
SKIP_EXIST=${SKIP_EXIST:-0}            # 1 = run 阶段跳过"已完成的同名文件"（行数 ≥ GSM_N）⇒ 重跑安全
PROGRESS=${PROGRESS:-5}                # 长跑一定要有进度，否则看不出是卡住还是在跑
TAU_PCT=${TAU_PCT:-90}
GAP_TAU=${GAP_TAU:-auto}               # gap_supp 的 τ 值：auto（按分位标定）/ 数值（如 0 = 死区消融）
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

# ---------------- smoke：小样本探针（默认 A + Bimf；先确认"哪组还能做 GSM8K"）----------------
if [ "$STAGE" = "smoke" ]; then
  SMOKE=../datasets/utility/gsm8k_smoke${SMOKE_N}.jsonl
  head -"$SMOKE_N" "$GSM" > "$SMOKE"
  echo "===== [smoke] $SMOKE_N 条 × $MT token | SMOKE_SCEN=$SMOKE_SCEN（method=vanilla）====="
  for s in $SMOKE_SCEN; do
    set -- $(scen_models "$s")
    prefix="$1"; m1="$2"; m2="$3"; m3="$4"
    if [ -z "$prefix" ]; then echo "跳过未知场景：$s（可选 A / Bif / Bhash / Bimf）"; continue; fi
    OUT="../outputs/smoke_gsm_${s}.jsonl"
    echo "--- 场景 $s: $m1 / $m2 / $m3 ---"
    t0=$(date +%s)
    run python ensemble_logit.py --test_set "$SMOKE" \
      --model_path1 "$m1" --model_path2 "$m2" --model_path3 "$m3" \
      --output_file "$OUT" --max_new_tokens "$MT" --method vanilla --progress_every 1
    t1=$(date +%s)
    d=$((t1 - t0))
    echo "⏱ $s: ${d}s / $SMOKE_N 条 ⇒ 100 条 ≈ $((d * 100 / SMOKE_N / 60)) min/次"
    python check_gsm_output.py "$OUT" --items "$SMOKE_N" --cap "$MT"
  done
fi

# ---------------- probe1：单模型对照（判断"是模型不会做，还是集成把它毁了"）----------------
if [ "$STAGE" = "probe1" ]; then
  SMOKE=../datasets/utility/gsm8k_smoke${SMOKE_N}.jsonl
  if [ ! -f "$SMOKE" ]; then head -"$SMOKE_N" "$GSM" > "$SMOKE"; fi
  echo "===== [probe1] 单模型（base / IF / Hash / ImF）在 $SMOKE_N 条上的 GSM8K ====="
  echo "注意：single_model_test.py 是**采样**解码（T=0.7, top_p=0.85），与集成的贪心解码不同 ⇒ 只看量级不看小数点"
  for m in "$BASE" "$M_IF" "$M_HASH" "$M_IMF"; do
    nm=$(basename "$m")
    OUT="../outputs/smoke1_gsm_${nm}.jsonl"
    echo "--- 单模型 $nm ---"
    run python single_model_test.py --test_set "$SMOKE" --model_path1 "$m" \
      --output_file "$OUT" --max_new_tokens "$MT"
    python check_gsm_output.py "$OUT" --items "$SMOKE_N" --cap "$MT"
  done
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
        gap_supp)
          extra="--gap_tau $GAP_TAU --gap_tau_pct $TAU_PCT --alpha $ALPHA --clean_path $CLEAN"
          # ⚠️ 消融档必须带档位后缀，否则**会覆盖 ⭐ 主档文件** p0_*_gap_supp_gsm.jsonl
          if [ "$GAP_TAU" != "auto" ]; then
            msuf="_gtau${GAP_TAU}a${ALPHA}"
          elif [ "$TAU_PCT" != "90" ] || [ "$ALPHA" != "1" ]; then
            msuf="_gtau${TAU_PCT}a${ALPHA}"
          fi
          ;;
        thresh_ours) extra="--tau_pct $TAU_PCT --clean_path $CLEAN" ;;
        temperature) extra="--T $T"; msuf="_T${T}" ;;
      esac
      outf="../outputs/${prefix}_${m}${msuf}_gsm.jsonl"
      if [ "$SKIP_EXIST" = "1" ] && [ -f "$outf" ] && [ "$(wc -l < "$outf" | tr -d ' ')" -ge "$GSM_N" ]; then
        echo "跳过（已完成，$(wc -l < "$outf" | tr -d ' ') 行）：$outf"
        continue
      fi
      run python ensemble_logit.py --test_set "$GSM" \
        --model_path1 "$m1" --model_path2 "$m2" --model_path3 "$m3" \
        --output_file "$outf" \
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
echo "然后写进 doc/数据_第二任务GSM8K.md §T7（表：方法 × {3fp, 1fp-ImF} 的 GSM8K ACC + 配对 Δ）"
