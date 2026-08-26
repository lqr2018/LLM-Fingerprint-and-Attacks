#########IF Qwen2.5-7B（适配 2×4090：ZeRO-3 + CPU offload + fp16）
# 消费级显卡无 NVLink，禁用 P2P 更稳定；若 CPU 内存紧张可注释下一行
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
deepspeed --master_port 29500 --num_gpus=2  ../Fingerprint_dataset/train_fingerprint.py \
--deepspeed ../Fingerprint_dataset/ds_config.json \
--model_name_or_path ../models/base/Qwen2.5-7B \
--data_path ../Fingerprint_dataset/IF/train_IF_60.json \
--output_dir ../models/fingerprint/IF_sft_Qwen2.5-7B \
--num_train_epochs 20 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 100 \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--report_to "tensorboard" \
--gradient_checkpointing True \
--fp16 True
# 从 TFA_SVA/ 目录执行：bash Fingerprint_dataset/run.sh
# 训练 Hash/ImF 时，把 --data_path / --output_dir 换成对应的
#   train_chain_hash60.json → ../models/fingerprint/Hash_sft_Qwen2.5-7B
#   train_stego60.json     → ../models/fingerprint/ImF_sft_Qwen2.5-7B
# 若服务器 CPU 内存 < 100GB，offload 可能 OOM，可改用 LoRA 方案（见复现指南）
