# -*- coding: utf-8 -*-
"""
集中配置：统一管理模型 / 数据集 / 输出路径。

所有脚本通过本文件读取默认路径，避免散落的硬编码路径。
目录约定（位于仓库根目录下）：

    models/
      base/                      # 基础模型（train_fingerprint.py 的训练输入）
        Qwen2.5-7B/
      fingerprint/               # 训练好的指纹模型（SVA/TFA 的受害者）
        IF_sft_Qwen2.5-7B/       #   → MODEL_PATH1
        Hash_sft_Qwen2.5-7B/     #   → MODEL_PATH2
        ImF_sft_Qwen2.5-7B/      #   → MODEL_PATH3

用法：
    命令行不传 --model_path* 时，脚本自动使用这里的默认路径；
    显式传参则覆盖默认值。
"""
from pathlib import Path

# ------------------------------------------------------------------
# 目录定位：config.py 位于 TFA_SVA/ 下，仓库根目录是其上一级
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

BASE_MODEL_DIR = MODELS_DIR / "base"                 # 基础模型
FINGERPRINT_MODEL_DIR = MODELS_DIR / "fingerprint"   # 训练好的指纹模型

# ------------------------------------------------------------------
# 模型路径（按你实际放在 models/ 下的文件名修改这里即可）
# ------------------------------------------------------------------
# 基础模型：train_fingerprint.py 的 --model_name_or_path 默认值
BASE_MODEL = (BASE_MODEL_DIR / "Qwen2.5-7B").as_posix()

# 三个指纹模型：SVA.py / TFA.py 的 --model_path1/2/3 默认值
MODEL_PATH1 = (FINGERPRINT_MODEL_DIR / "IF_sft_Qwen2.5-7B").as_posix()
MODEL_PATH2 = (FINGERPRINT_MODEL_DIR / "Hash_sft_Qwen2.5-7B").as_posix()
MODEL_PATH3 = (FINGERPRINT_MODEL_DIR / "ImF_sft_Qwen2.5-7B").as_posix()

# 单模型评测默认模型（single_model_test.py / GRI_attack.py 用）
DEFAULT_TEST_MODEL = MODEL_PATH1

# ------------------------------------------------------------------
# 数据集与输出目录（可选，按需使用）
# ------------------------------------------------------------------
DATA_DIR = (REPO_ROOT / "datasets").as_posix()
OUTPUT_DIR = (REPO_ROOT / "outputs").as_posix()
