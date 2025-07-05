# GPT-2 Fine-Tuning Demo

This project demonstrates how to fine-tune the GPT-2 language model on a small custom text dataset and then generate new text from a prompt.

## How to run

1. Install dependencies:

    pip install transformers torch

2. Place your training data in `train.txt`.

3. Run the fine-tuning script:

    python gpt2_finetune.py

4. After training, it will generate text from a sample prompt.