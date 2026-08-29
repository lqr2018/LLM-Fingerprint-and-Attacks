#!/bin/bash
# 阈值门控实验（P4b）：thresh_ours 在 τ=85/90/95 下的 ASR + ARC
# 目的：验证"只对超过 Clean 正常范围的 disagreement 抑制"能否减少正常能力损失（ARC 回升）
# 从 TFA_SVA/ 执行：bash run_thresh.sh
cd "$(dirname "$0")"

# 固定 α=1.0，扫 τ 三档
for p in 85 90 95; do
  echo "===== thresh_ours tau_pct=$p | IF fingerprint ASR ====="
  python ensemble_logit.py --test_set ../datasets/fingerprint_test/test_IF_10.json \
    --output_file ../outputs/ens_thresh${p}_if.jsonl --max_new_tokens 40 \
    --method thresh_ours --tau_pct $p --alpha 1.0

  echo "===== thresh_ours tau_pct=$p | ARC Utility ====="
  python ensemble_logit.py --test_set ../datasets/utility/arc_100.jsonl \
    --output_file ../outputs/ens_thresh${p}_arc.jsonl --max_new_tokens 32 \
    --method thresh_ours --tau_pct $p --alpha 1.0
done

echo "===== all done ====="
