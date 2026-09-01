#!/bin/bash
# 新设置：1 个注入指纹的模型 + 2 个干净(base)模型
# 目的：更贴近现实中"ensemble 混合有/无指纹模型"的场景，观察各方法表现
# 从 TFA_SVA/ 执行：bash run_ensemble_1fp.sh

# ==================== 可修改参数 ====================
ALPHA=1.0                              # ← 抑制强度（ours/thresh_ours/random 用）
BASE_MODEL="../models/base/Qwen2.5-7B" # ← 干净模型路径
# ====================================================

cd "$(dirname "$0")"

run_group() {
  local fp_model="$1"   # 指纹模型路径
  local test_set="$2"   # 测试集
  local tag="$3"        # 输出标签
  local mt="$4"         # max_new_tokens
  for m in vanilla ours median temperature clipping confidence random; do
    echo "--- [$tag] method: $m (alpha=$ALPHA) ---"
    python ensemble_logit.py --test_set "$test_set" \
      --model_path1 "$fp_model" \
      --model_path2 "$BASE_MODEL" --model_path3 "$BASE_MODEL" \
      --output_file "../outputs/ens1fp_${m}_${tag}_a${ALPHA}.jsonl" --max_new_tokens "$mt" \
      --method $m --alpha "$ALPHA"
  done
  # thresh_ours：⚠️ 不传 --tau，让代码用"当前 1fp 模型组合"在 Clean 数据上重新算 τ
  #（1fp 组合的 disagreement 分布与 3 指纹组合不同，不能直接沿用之前的 τ 值）
  echo "--- [$tag] method: thresh_ours (auto tau from 1fp Clean) ---"
  python ensemble_logit.py --test_set "$test_set" \
    --model_path1 "$fp_model" \
    --model_path2 "$BASE_MODEL" --model_path3 "$BASE_MODEL" \
    --output_file "../outputs/ens1fp_thresh_${tag}_a${ALPHA}.jsonl" --max_new_tokens "$mt" \
    --method thresh_ours --tau_pct 90 --alpha "$ALPHA"
}

echo "===== 1×IF指纹 + 2×base，测 IF 测试集 ====="
run_group "../models/fingerprint/IF_sft_Qwen2.5-7B" \
  "../datasets/fingerprint_test/test_IF_10.json" "if" 40

echo "===== 1×Hash指纹 + 2×base，测 Hash 测试集 ====="
run_group "../models/fingerprint/Hash_sft_Qwen2.5-7B" \
  "../datasets/fingerprint_test/test_chain_hash10.json" "hash" 40

echo "===== 1×ImF指纹 + 2×base，测 ImF 测试集 ====="
run_group "../models/fingerprint/ImF_sft_Qwen2.5-7B" \
  "../datasets/fingerprint_test/test_stego10.jsonl" "stego" 128

echo "===== all done ====="

