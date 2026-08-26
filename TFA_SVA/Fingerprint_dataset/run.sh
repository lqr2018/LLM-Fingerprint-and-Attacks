#########IF Qwen2.5-7B（适配 2×4090：ZeRO-3 + CPU offload + fp16）
# 必须在本目录（TFA_SVA/Fingerprint_dataset/）执行：bash run.sh
# 原因：train_fingerprint.py 里 import utils 依赖本目录的 utils.py
# 说明：本机之前用默认 NCCL 配置成功训练过，故不设置 NCCL_* 环境变量
# （之前尝试 NCCL_P2P_DISABLE/NCCL_SHM_DISABLE/NCCL_SOCKET_IFNAME=lo 反而在 all-gather 大通信时异常）
# 显存碎片优化（24GB 4090 建议开启）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
deepspeed --master_port 29500 --num_gpus=2  train_fingerprint.py \
--deepspeed ds_config.json \
--model_name_or_path ../../models/base/Qwen2.5-7B \
--data_path IF/train_IF_60.json \
--output_dir ../../models/fingerprint/IF_sft_Qwen2.5-7B \
--num_train_epochs 20 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "no" \
--save_total_limit 1 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--gradient_checkpointing True \
--fp16 True \
--report_to "none"
# 若想用 TensorBoard 看训练曲线：装 tensorboard 后把 --report_to "none" 改成 --report_to "tensorboard"
# 训练 Hash/ImF 时，把 --data_path / --output_dir 换成对应的：
#   IF/train_IF_60.json            → ../../models/fingerprint/IF_sft_Qwen2.5-7B
#   Hash/train_chain_hash60.json   → ../../models/fingerprint/Hash_sft_Qwen2.5-7B
#   ImF/train_stego60.json         → ../../models/fingerprint/ImF_sft_Qwen2.5-7B
# 若服务器 CPU 内存 < 100GB，offload 可能 OOM，可改用 LoRA 方案（见复现指南）
