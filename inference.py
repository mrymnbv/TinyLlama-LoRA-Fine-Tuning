"""
inference.py

This file is used to test the fine-tuned TinyLlama model.

It loads the trained LoRA adapter and generates
responses for custom prompts.
"""


import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel


from config import (
    MODEL_NAME,
    ADAPTER_OUTPUT_DIR,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P
)



# LOAD MODEL AND ADAPTER

def load_fine_tuned_model():
    """
    Loads the base TinyLlama model and applies
    the trained LoRA adapter.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        ADAPTER_OUTPUT_DIR
    )

    tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto"
    )


    model = PeftModel.from_pretrained(
        model,
        ADAPTER_OUTPUT_DIR
    )


    return model, tokenizer



# TEXT GENERATION FUNCTION

def generate(
    model,
    tokenizer,
    prompt
):
    """
    Generates a response from the fine-tuned model.

    Args:
        model: Fine-tuned language model
        tokenizer: Model tokenizer
        prompt: Input text

    Returns:
        Generated response text
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)


    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P
        )


    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )



# TEST EXAMPLES


if __name__ == "__main__":


    model, tokenizer = load_fine_tuned_model()


    print("\nTEST 1:")
    print(
        generate(
            model,
            tokenizer,
            "What is the capital of France?"
        )
    )


    print("\nTEST 2:")
    print(
        generate(
            model,
            tokenizer,
            "What is 15 x 6?"
        )
    )


    print("\nTEST 3:")
    print(
        generate(
            model,
            tokenizer,
            "Who wrote Hamlet?"
        )
    )
