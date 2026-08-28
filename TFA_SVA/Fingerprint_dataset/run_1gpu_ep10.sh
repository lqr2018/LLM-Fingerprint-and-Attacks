#########IF Qwen2.5-7B 单卡训练 —— epoch=10 快速验证版（fingerprint strength 验证）
# 目的：检查降低训练强度（20→10 epoch）能否在 ASR 仍≈1.0 时减少 Utility 损失（ARC 回升）
# 输出到 _ep10 独立目录，不覆盖 epoch=20 的模型
# 必须在本目录（TFA_SVA/Fingerprint_dataset/）执行：bash run_1gpu_ep10.sh
export CUDA_VISIBLE_DEVICES=0
# 显存碎片优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
deepspeed --master_port 29500 train_fingerprint.py \
--deepspeed ds_config.json \
--model_name_or_path ../../models/base/Qwen2.5-7B \
--data_path IF/train_IF_60.json \
--output_dir ../../models/fingerprint/IF_sft_Qwen2.5-7B_ep10 \
--num_train_epochs 10 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 1 \
--evaluation_strategy "no" \
--save_strategy "no" \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--gradient_checkpointing True \
--fp16 True \
--report_to "none"
# 训练 Hash/ImF 的 ep10 版本时，把 --data_path / --output_dir 换成对应的：
#   Hash/train_chain_hash60.json   → ../../models/fingerprint/Hash_sft_Qwen2.5-7B_ep10
#   ImF/train_stego60.json         → ../../models/fingerprint/ImF_sft_Qwen2.5-7B_ep10