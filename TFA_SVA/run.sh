# SVA（句子级投票攻击）——模型路径默认从 config.py 读取，无需传 --model_path*
# 注意：仓库实际脚本名是 SVA.py（不是 SVA_3.py）；--test_set 与值之间必须有空格
python SVA.py \
    --test_set /你的路径/fingerprint_test.jsonl \
    --output_file ../outputs/out_sva.jsonl \
    --per_device_batch_size 1 \
    --max_new_tokens 64

# TFA（词元级融合攻击）
python TFA.py \
    --test_set /你的路径/fingerprint_test.jsonl \
    --output_file ../outputs/out_tfa.jsonl \
    --per_device_batch_size 1 \
    --max_new_tokens 64

# 若要临时换模型，可追加参数覆盖默认值，例如：
#   python SVA.py --test_set ... --model_path1 ../models/fingerprint/IF_sft_Qwen2.5-7B ...


