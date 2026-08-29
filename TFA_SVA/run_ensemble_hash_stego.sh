#!/bin/bash
# 批量跑 Hash 和 ImF 指纹的 7 种融合方法（3 卡机器，从 TFA_SVA/ 执行：bash run_ensemble_hash_stego.sh）
# ARC(Utility) 之前已跑过，这里只跑 fingerprint ASR
cd "$(dirname "$0")"

echo "===== Hash 指纹（10 条子集，短答案）====="
for m in vanilla ours median temperature clipping confidence random; do
  echo "--- method: $m ---"
  python ensemble_logit.py --test_set ../datasets/fingerprint_test/test_chain_hash10.json \
    --output_file ../outputs/ens_${m}_hash.jsonl --max_new_tokens 40 --method $m
done

echo "===== ImF 指纹（10 条子集，长句）====="
for m in vanilla ours median temperature clipping confidence random; do
  echo "--- method: $m ---"
  python ensemble_logit.py --test_set ../datasets/fingerprint_test/test_stego10.jsonl \
    --output_file ../outputs/ens_${m}_stego.jsonl --max_new_tokens 128 --method $m
done

echo "===== all done ====="
