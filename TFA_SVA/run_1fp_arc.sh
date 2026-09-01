#!/bin/bash
# 1fp 设置（1 指纹 + 2 base）的 ARC(Utility) 评测
# ⚠️ 每组模型组合不同（model_path1 分别 = IF/Hash/ImF 指纹模型），ARC 必须按组单独测
# 从 TFA_SVA/ 执行：bash run_1fp_arc.sh

# ==================== 可修改参数 ====================
ALPHA=1.0                              # ← 抑制强度
BASE_MODEL="../models/base/Qwen2.5-7B" # ← 干净模型
# ====================================================

cd "$(dirname "$0")"

run_arc_group() {
  local fp_model="$1"   # 该组的指纹模型
  local tag="$2"        # 输出标签
  for m in vanilla ours thresh_ours median temperature clipping confidence random; do
    echo "--- [${tag}] ARC method: $m (alpha=$ALPHA) ---"
    python ensemble_logit.py --test_set ../datasets/utility/arc_100.jsonl \
      --model_path1 "$fp_model" \
      --model_path2 "$BASE_MODEL" --model_path3 "$BASE_MODEL" \
      --output_file "../outputs/ens1fp_arc_${m}_${tag}_a${ALPHA}.jsonl" --max_new_tokens 32 \
      --method $m --alpha "$ALPHA" --tau_pct 90
  done
}

echo "===== ARC | 1×IF指纹 + 2×base ====="
run_arc_group "../models/fingerprint/IF_sft_Qwen2.5-7B" "if"

echo "===== ARC | 1×Hash指纹 + 2×base ====="
run_arc_group "../models/fingerprint/Hash_sft_Qwen2.5-7B" "hash"

echo "===== ARC | 1×ImF指纹 + 2×base ====="
run_arc_group "../models/fingerprint/ImF_sft_Qwen2.5-7B" "stego"

echo "===== all done ====="
