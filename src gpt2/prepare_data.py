import json
import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE), "data", "support_tickets.json")
MODEL_NAME = "gpt2"
MAX_LEN = 256

def load_squad_like(path):
    with open(path, "r") as f:
        data = json.load(f)
    qa_list = []
    for doc in data.get("data", []):
        for para in doc.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                # For GPT-2 we just keep context, question, and answer text
                answer_text = qa.get("answers", [{}])[0].get("text", "")
                qa_list.append({
                    "id": qa.get("id"),
                    "input_text": f"Context: {context}\nQuestion: {qa.get('question')}\nAnswer:",
                    "target_text": answer_text
                })
    return qa_list

def tokenize_and_format(qa_list):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Fix: GPT-2 has no pad token -> set it to eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(example):
        # Tokenize input (context + question)
        inputs = tokenizer(
            example["input_text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length"
        )
        # Tokenize output (answer)
        labels = tokenizer(
            example["target_text"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length"
        )

        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized_dataset = Dataset.from_list(qa_list).map(preprocess, batched=False)
    return tokenized_dataset

def build_dataset():
    qa_list = load_squad_like(DATA_PATH)
    # Split train/val
    split_idx = int(0.9 * len(qa_list)) if len(qa_list) > 1 else len(qa_list)
    train = qa_list[:split_idx]
    val = qa_list[split_idx:]
    train_ds = tokenize_and_format(train)
    val_ds = tokenize_and_format(val)
    return DatasetDict({"train": train_ds, "validation": val_ds})

if __name__ == "__main__":
    ds = build_dataset()
    print(ds)
