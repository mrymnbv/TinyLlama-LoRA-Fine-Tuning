"""
data.py

This file handles dataset creation and preprocessing.

The workflow includes:
- Creating an instruction-response dataset
- Formatting examples into prompts
- Tokenizing the text
- Preparing the dataset for PyTorch training
"""


from datasets import Dataset


# CREATE DATASET

def create_dataset():
    """
    Creates a small instruction-response dataset
    for supervised fine-tuning.

    Returns:
        Dataset object containing training examples.
    """

    examples = [
        {
            "instruction": "What is the capital of France?",
            "response": "Paris"
        },
        {
            "instruction": "Who wrote Hamlet?",
            "response": "William Shakespeare"
        },
        {
            "instruction": "Translate 'Good morning' to Spanish.",
            "response": "Buenos días"
        },
        {
            "instruction": "What is 12 x 9?",
            "response": "108"
        },
        {
            "instruction": "Name a primary color.",
            "response": "Red"
        },
    ]

    dataset = Dataset.from_list(examples)

    print("Dataset loaded:")
    print(dataset)

    return dataset



# FORMAT TRAINING PROMPTS

def format_example(example):
    """
    Converts each instruction-response pair into a
    structured prompt format for the language model.
    """

    text = f"""### Instruction:
{example['instruction']}

### Response:
{example['response']}"""

    return {
        "text": text
    }




# TOKENIZATION

def tokenize_dataset(dataset, tokenizer):
    """
    Tokenizes the dataset and prepares labels
    for causal language model training.

    Args:
        dataset: Hugging Face Dataset object
        tokenizer: Model tokenizer

    Returns:
        Tokenized dataset ready for training.
    """

    MAX_LENGTH = 256


    def tokenize(example):

        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

        # For causal language modeling,
        # labels are the same as input tokens
        tokens["labels"] = tokens["input_ids"].copy()

        return tokens


    # Apply prompt formatting
    dataset = dataset.map(format_example)


    print("\nExample formatted:\n")
    print(dataset[0]["text"])


    # Apply tokenization
    tokenized_dataset = dataset.map(tokenize)


    # Remove original text fields because
    # the model only needs tokenized inputs
    tokenized_dataset = tokenized_dataset.remove_columns(
        [
            "instruction",
            "response",
            "text"
        ]
    )


    # Convert dataset format to PyTorch tensors
    tokenized_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels"
        ]
    )


    print("\nTokenization complete.")
    print(tokenized_dataset[0])


    return tokenized_dataset




# COMPLETE DATA PREPARATION PIPELINE

def prepare_dataset(tokenizer):
    """
    Runs the complete preprocessing pipeline.

    Steps:
    1. Create dataset
    2. Format prompts
    3. Tokenize examples
    4. Convert to PyTorch format
    """

    dataset = create_dataset()

    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer
    )

    return tokenized_dataset
