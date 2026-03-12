from transformers import BertTokenizer, BertForSequenceClassification, \
    Trainer, TrainingArguments
from datasets import Dataset
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(jsonl_path: str) -> tuple[list[str], list[int]]:
    """
    Load text data and labels from a prepared JSONL file. The JSONL file is
    expected to have a 'text' field for the raw text and a 'tokens' field for
    the tokenized version. For demonstration purposes, this function generates
    dummy binary labels based on the length of the text (even vs. odd length).

    Parameters
    ----------
    jsonl_path : str
        Path to the prepared JSONL file.

    Returns
    -------
    tuple[list[str], list[int]]
        A tuple containing a list of texts and a list of corresponding labels.
    """

    texts, labels = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(len(obj["text"]) % 2)  # Dummy label
    return texts, labels


def run_bert_baseline(jsonl_path: str) -> None:
    """
    Run a BERT baseline on the prepared dataset. The dataset is loaded from a
    JSONL file, tokenized using a BERT tokenizer, and then used to train and
    evaluate a BERT model.

    Parameters
    ----------
    jsonl_path : str
        Path to the prepared JSONL file containing the dataset.
    """

    texts, labels = load_data(jsonl_path)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    train_dataset = Dataset.from_dict(
        {"text": X_train, "label": y_train}
    ).map(tokenize, batched=True)
    test_dataset = Dataset.from_dict(
        {"text": X_test, "label": y_test}
    ).map(tokenize, batched=True)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    training_args = TrainingArguments(
        output_dir="./models/bert-baseline",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=10,
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()
    preds = trainer.predict(test_dataset)
    y_pred = np.argmax(preds.predictions, axis=1)
    print("BERT Baseline Results:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description="Run BERT baseline on prepared data."
    )
    parser.add_argument(
        "--prepared_jsonl",
        type=str,
        required=True,
        help="Path to prepared JSONL data"
    )
    args = parser.parse_args()
    run_bert_baseline(args.prepared_jsonl)
