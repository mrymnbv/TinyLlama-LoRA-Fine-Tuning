"""
config.py

This file contains the main configuration values
used for model training and LoRA fine-tuning.
"""



# MODEL CONFIGURATION

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# DATA CONFIGURATION

MAX_LENGTH = 256



# LORA CONFIGURATION

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
]



# TRAINING CONFIGURATION

OUTPUT_DIR = "./tinyllama-lora"

ADAPTER_OUTPUT_DIR = "./tinyllama-lora-adapter"


PER_DEVICE_TRAIN_BATCH_SIZE = 2

GRADIENT_ACCUMULATION_STEPS = 2

LEARNING_RATE = 2e-4

NUM_EPOCHS = 3

LOGGING_STEPS = 1

SAVE_STRATEGY = "epoch"

WARMUP_STEPS = 10



# INFERENCE CONFIGURATION

MAX_NEW_TOKENS = 80

TEMPERATURE = 0.7

TOP_P = 0.9
