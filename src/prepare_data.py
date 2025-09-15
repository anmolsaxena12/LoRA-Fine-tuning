import json
import os
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE), "data", "support_tickets.json")
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256

def load_squad_like(path):
    with open(path, "r") as f:
        data = json.load(f)
    qa_list = []
    for doc in data.get("data", []):
        for para in doc.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                qa_list.append({
                    "id": qa.get("id"),
                    "question": qa.get("question"),
                    "context": context,
                    "answers": qa.get("answers", [])
                })
    return qa_list

def tokenize_and_align(qa_list):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(example):
        tokenized = tokenizer(
            example["question"],
            example["context"],
            truncation="only_second",
            max_length=MAX_LEN,
            padding="max_length",
            return_offsets_mapping=True
        )
        # Compute start and end positions
        answers = example["answers"]
        if answers:
            start_char = answers[0]["answer_start"]
            end_char = start_char + len(answers[0]["text"])
            offsets = tokenized["offset_mapping"]
            sequence_ids = tokenized.sequence_ids()
            context_start = sequence_ids.index(1)
            context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

            token_start = context_start
            token_end = context_end

            for idx, (start, end) in enumerate(offsets):
                if start <= start_char < end:
                    token_start = idx
                if start < end_char <= end:
                    token_end = idx
                    break
        else:
            token_start, token_end = 0, 0

        tokenized["start_positions"] = token_start
        tokenized["end_positions"] = token_end
        tokenized.pop("offset_mapping")
        return tokenized

    tokenized_dataset = Dataset.from_list(qa_list).map(preprocess)
    return tokenized_dataset

def build_dataset():
    qa_list = load_squad_like(DATA_PATH)
    train = qa_list[:-1]
    val = qa_list[-1:]
    train_ds = tokenize_and_align(train)
    val_ds = tokenize_and_align(val)
    return DatasetDict({"train": train_ds, "validation": val_ds})

if __name__ == "__main__":
    ds = build_dataset()
    print(ds)
