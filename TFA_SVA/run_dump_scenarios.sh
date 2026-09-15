#!/bin/bash
# P1-3：为"跨场景离线筛选"补 dump（只做诊断输出，不改变任何实验结果）
#
# 用法（在 TFA_SVA/ 下）：
#   bash run_dump_scenarios.sh            # 全部 5 次
#   bash run_dump_scenarios.sh 1fp        # 只跑 1fp（Hash / IF）
#   bash run_dump_scenarios.sh 3fp        # 只跑 3fp（IF / Hash / ImF 三个测试集）
#   DRYRUN=1 bash run_dump_scenarios.sh   # 只打印命令
#
# 每次产出两个文件（写到 ../outputs/）：
#   <name>.jsonl.topn.jsonl    # 逐步 vanilla top-200 坐标的 各模型 logit（离线筛选用，仅存不改变结果）
#   <name>.jsonl.debug.jsonl   # 逐步坐标级诊断（max_delta / penalty / margin / flip）
# 注意：ImF-1fp 已有一份（output/dbg_imf_ours.jsonl*），本脚本不重复跑。

cd "$(dirname "$0")"
STAGE=${1:-all}
RUNS=0

MT_SHORT=40      # IF / Hash 指纹答案是单个词
MT_LONG=128      # ImF 答案是一整句
DTOP=5           # 诊断 top-k
DLOG=200         # dump 的 top-N（实测 k≥20 即与全词表一致；200 为保守取值）

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

dump() {   # $1=name  $2=test_set  $3=maxtok  $4=m1  $5=m2  $6=m3
  run python ensemble_logit.py --test_set "$2" \
    --model_path1 "$4" --model_path2 "$5" --model_path3 "$6" \
    --output_file "../outputs/$1.jsonl" --max_new_tokens "$3" \
    --method ours --debug_topk "$DTOP" --debug_dump_logits "$DLOG"
}

if [ "$STAGE" = "1fp" ] || [ "$STAGE" = "all" ]; then
  echo "===== 1fp（1 指纹 + 2×base）====="
  dump dumpB_hash "$FP_HASH" "$MT_SHORT" "$M_HASH" "$BASE" "$BASE"
  dump dumpB_if   "$FP_IF"   "$MT_SHORT" "$M_IF"   "$BASE" "$BASE"
fi

if [ "$STAGE" = "3fp" ] || [ "$STAGE" = "all" ]; then
  echo "===== 3fp（IF + Hash + ImF 三模型）====="
  dump dumpA_if   "$FP_IF"   "$MT_SHORT" "$M_IF" "$M_HASH" "$M_IMF"
  dump dumpA_hash "$FP_HASH" "$MT_SHORT" "$M_IF" "$M_HASH" "$M_IMF"
  dump dumpA_imf  "$FP_IMF"  "$MT_LONG"  "$M_IF" "$M_HASH" "$M_IMF"
fi

echo ""
echo "===== [stage=$STAGE] 共 $RUNS 次 dump 运行 ====="
echo "产出的 *.topn.jsonl 请放回本地 output/；本地筛选命令："
echo "  python TFA_SVA/offline_gate_screen_batch.py --glob \"output/*.topn.jsonl\""
