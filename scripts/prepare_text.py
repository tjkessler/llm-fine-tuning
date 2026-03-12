import argparse
import json
import os
import re
import time

from tqdm import tqdm
from transformers import AutoTokenizer


def clean_text(text: str) -> str:
    """
    Clean the input text by removing non-printable characters, collapsing
    multiple spaces/newlines, and stripping leading/trailing whitespace.

    Parameters
    ----------
    text : str
        The raw text to clean.

    Returns
    -------
    str
        The cleaned text.
    """

    # Remove non-printable characters
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    # Replace multiple spaces/newlines with single
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()


def main(data_dir: str, output_jsonl: str, base_model: str = "gpt2") -> None:
    """
    Prepare raw text files for training by cleaning and tokenizing them, then
    saving as a JSONL file with 'text' and 'tokens' fields.

    Parameters
    ----------
    data_dir : str
        Directory containing the raw text files.
    output_jsonl : str
        Output file path for the cleaned and tokenized dataset in JSONL format.
    base_model : str, optional
        Base model to use for tokenization (default: "gpt2").
    """

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    _t_start = time.time()
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for fname in tqdm(files, desc="Processing files"):
            fpath = os.path.join(data_dir, fname)
            with open(fpath, "r", encoding="utf-8") as in_f:
                raw_text = in_f.read()
            cleaned = clean_text(raw_text)
            # Tokenize (returns list of token ids)
            tokens = tokenizer.encode(cleaned)
            # Write as JSONL: {"text": ..., "tokens": ...}
            out_f.write(json.dumps({"text": cleaned, "tokens": tokens}) + "\n")
    _t_end = time.time()
    print(f"Processed {len(files)} files in {_t_end - _t_start:.2f} seconds. Output: {output_jsonl}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prepare Europe PMC data for training.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the raw text files.")
    parser.add_argument("--output_jsonl", type=str, required=True, help="Output file for cleaned/tokenized data.")
    parser.add_argument("--base_model", type=str, default="gpt2", help="Base model for tokenization.")
    args = parser.parse_args()
    main(args.data_dir, args.output_jsonl, args.base_model)
