from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

def main():
    # Load GPT-2 base model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load training data
    dataset_file = "train.txt"
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_file,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Training settings
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=100,
        save_total_limit=1,
        logging_steps=50
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # Train
    trainer.train()
    trainer.save_model("./gpt2-finetuned")

    # Generate text from prompt
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=50,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n=== GENERATED TEXT ===\n")
    print(generated_text)

if __name__ == "__main__":
    main()