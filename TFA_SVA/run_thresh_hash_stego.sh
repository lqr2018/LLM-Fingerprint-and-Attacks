#!/bin/bash
# 带阈值的 C&H(Hash) 和 ImF 阈值门控实验
# 阈值已用 Clean 数据算好（τ85/τ90/τ95），通过 --tau 直接传入，跳过重算
# 从 TFA_SVA/ 执行：bash run_thresh_hash_stego.sh

# ==================== 可修改参数 ====================
ALPHA=1.0                              # ← 抑制强度，直接改这里即可
TAUS=(0.9517 1.2036 1.7158)            # ← τ85 / τ90 / τ95（对应 Clean 数据的 P85/P90/P95）
# ====================================================

cd "$(dirname "$0")"

# ---- Hash 指纹（10 条子集，短答案）----
for t in "${TAUS[@]}"; do
  echo "===== Hash | thresh_ours tau=$t alpha=$ALPHA ====="
  python ensemble_logit.py --test_set ../datasets/fingerprint_test/test_chain_hash10.json \
    --output_file "../outputs/ens_thresh_hash_t${t}_a${ALPHA}.jsonl" --max_new_tokens 40 \
    --method thresh_ours --tau "$t" --alpha "$ALPHA"
done

# ---- ImF 指纹（10 条子集，长句）----
for t in "${TAUS[@]}"; do
  echo "===== ImF | thresh_ours tau=$t alpha=$ALPHA ====="
  python ensemble_logit.py --test_set ../datasets/fingerprint_test/test_stego10.jsonl \
    --output_file "../outputs/ens_thresh_stego_t${t}_a${ALPHA}.jsonl" --max_new_tokens 128 \
    --method thresh_ours --tau "$t" --alpha "$ALPHA"
done

echo "===== all done ====="
