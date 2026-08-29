#!/bin/bash
# 新设置：1 个注入指纹的模型 + 2 个干净(base)模型
# 目的：更贴近现实中"ensemble 混合有/无指纹模型"的场景，观察各方法表现
# 从 TFA_SVA/ 执行：bash run_ensemble_1fp.sh
cd "$(dirname "$0")"

BASE_MODEL="../models/base/Qwen2.5-7B"

run_group() {
  local fp_model="$1"   # 指纹模型路径
  local test_set="$2"   # 测试集
  local tag="$3"        # 输出标签
  local mt="$4"         # max_new_tokens
  for m in vanilla ours median temperature clipping confidence random; do
    echo "--- [$tag] method: $m ---"
    python ensemble_logit.py --test_set "$test_set" \
      --model_path1 "$fp_model" \
      --model_path2 "$BASE_MODEL" --model_path3 "$BASE_MODEL" \
      --output_file "../outputs/ens1fp_${m}_${tag}.jsonl" --max_new_tokens "$mt" --method $m
  done
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
