import argparse
import time

from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch


def main(
        data_path: str,
        output_dir: str,
        base_model: str = "gpt2",
        num_train_epochs: int = 3,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_seq_length: int = 1024,
     ) -> None:
    """
    Fine-tune a language model on a pre-tokenized dataset.

    Parameters
    ----------
    data_path : str
        Path to the prepared JSONL dataset with pre-tokenized 'tokens' field.
    output_dir : str
        Directory to save the fine-tuned model and tokenizer.
    base_model : str, optional
        Base model to fine-tune (default: "gpt2").
    num_train_epochs : int, optional
        Number of training epochs (default: 3).
    temperature : float, optional
        Sampling temperature for generation (default: 1.0).
    top_p : float, optional
        Nucleus sampling top_p for generation (default: 1.0).
    max_seq_length : int, optional
        Maximum sequence length for training and generation (default: 1024).
    """

    # Load dataset
    _t_start = time.time()
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"Loaded dataset with {len(dataset)} examples in {time.time() - _t_start:.2f} seconds.")

    # Load tokenizer and model
    _t_start = time.time()
    print("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    # Set pad_token if not present (GPT-2 has no pad_token by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    print(f"Loaded model and tokenizer in {time.time() - _t_start:.2f} seconds.")

    # Use pre-tokenized 'tokens' from the dataset
    print("Preparing batches from pre-tokenized 'tokens' field...")

    def data_collator(features, max_seq_length=max_seq_length):
        # Truncate sequences to max_seq_length, then pad
        truncated_features = []
        for f in features:
            tokens = f["tokens"][:max_seq_length]
            truncated_features.append({"tokens": tokens})
        max_length = max(len(f["tokens"]) for f in truncated_features)
        input_ids = []
        attention_mask = []
        for f in truncated_features:
            tokens = f["tokens"]
            pad_len = max_length - len(tokens)
            input_ids.append(torch.tensor(tokens + [tokenizer.pad_token_id] * pad_len, dtype=torch.long))
            attention_mask.append(torch.tensor([1] * len(tokens) + [0] * pad_len, dtype=torch.long))
        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(input_ids).clone(),
        }
        return batch

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=2,
        save_steps=500,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    _t_start = time.time()
    print("Starting training...")
    trainer.train()
    print(f"Training completed in {time.time() - _t_start:.2f} seconds.")
    trainer.save_model(output_dir)


    # Save tokenizer to output directory
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

    # --- Evaluation: Perplexity and Generation Example ---
    print("\nEvaluating model on a sample of the training data (perplexity)...")
    sample = dataset.select(range(min(100, len(dataset))))
    model.eval()
    device = next(model.parameters()).device
    losses = []
    with torch.no_grad():
        for ex in sample:
            input_ids = torch.tensor([ex["tokens"][:512]], dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_ids).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            # loss is averaged over all tokens in the sequence
            losses.append(outputs.loss.item())
    mean_loss = np.mean(losses)
    perplexity = np.exp(mean_loss)
    print(f"Mean loss: {mean_loss:.4f} | Perplexity: {perplexity:.2f}")

    # --- Qualitative: Text Generation Example ---
    print("\nText generation example:")
    prompt = "Tree-based machine learning"
    print(f"Prompt: {repr(prompt)}\n---")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated text:\n{generated}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune a model on tokenized data.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the prepared JSONL dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model to fine-tune (default: gpt2).")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs (default: 3).")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for generation (default: 1.0).")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus sampling top_p for generation (default: 1.0).")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for training and generation (default: 1024).")
    args = parser.parse_args()
    main(
        args.data_path,
        args.output_dir,
        args.base_model,
        args.num_train_epochs,
        args.temperature,
        args.top_p,
        args.max_seq_length,
    )
