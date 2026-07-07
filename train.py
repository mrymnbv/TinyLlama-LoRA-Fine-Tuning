"""
train.py

Main training script for TinyLlama fine-tuning using LoRA.

This script:
1. Loads the TinyLlama model
2. Prepares the dataset
3. Applies LoRA adapters
4. Fine-tunes the model
5. Saves the trained adapter
"""


import torch

from transformers import TrainingArguments

from trl import SFTTrainer

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)


from model import load_model

from data import prepare_dataset


from config import (
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,

    OUTPUT_DIR,
    ADAPTER_OUTPUT_DIR,

    PER_DEVICE_TRAIN_BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    NUM_EPOCHS,
    LOGGING_STEPS,
    SAVE_STRATEGY,
    WARMUP_STEPS
)




# LOAD MODEL AND TOKENIZER

model, tokenizer = load_model()




# PREPARE DATASET

tokenized_dataset = prepare_dataset(tokenizer)



# LORA CONFIGURATION

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=TARGET_MODULES
)


# Add LoRA adapters to the model
model = get_peft_model(
    model,
    lora_config
)


print("\nTrainable parameters:")
model.print_trainable_parameters()




# TRAINING CONFIGURATION

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=
        PER_DEVICE_TRAIN_BATCH_SIZE,

    gradient_accumulation_steps=
        GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,

    num_train_epochs=NUM_EPOCHS,

    logging_steps=LOGGING_STEPS,

    save_strategy=SAVE_STRATEGY,

    bf16=torch.cuda.is_available(),

    fp16=False,

    warmup_steps=WARMUP_STEPS,

    report_to="none"
)



# TRAINER SETUP

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
)



# START TRAINING

print("\nStarting training...")

trainer.train()



# SAVE TRAINED ADAPTER


trainer.save_model(
    ADAPTER_OUTPUT_DIR
)

tokenizer.save_pretrained(
    ADAPTER_OUTPUT_DIR
)


print(
    "\nTraining complete and model saved."
)
