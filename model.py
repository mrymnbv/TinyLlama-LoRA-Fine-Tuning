"""
model.py

This file handles loading the TinyLlama model and tokenizer.
The model is loaded with 4-bit quantization and prepared
for LoRA fine-tuning.
"""

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import prepare_model_for_kbit_training



# DEVICE CHECK
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)



# MODEL CONFIGURATION

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# MODEL LOADING FUNCTION

def load_model():

    """
    Loads TinyLlama with 4-bit quantization and prepares
    it for LoRA fine-tuning.

    Returns:
        model: Loaded language model
        tokenizer: Model tokenizer
    """


    # 4-bit quantization settings

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


    # Load tokenizer


    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token


    
    # Load quantized model

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"
    )


    
    # Prepare model for LoRA

    model = prepare_model_for_kbit_training(model)


    print("Model loaded and ready.")


    return model, tokenizer
