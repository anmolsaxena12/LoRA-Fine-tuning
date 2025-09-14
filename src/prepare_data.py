import json
from datasets import Dataset, DatasetDict
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE), "data", "support_tickets.json")

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

def build_dataset():
    qa_list = load_squad_like(DATA_PATH)
    # Split into train/validation in a simple deterministic way
    train = qa_list[:-1]
    val = qa_list[-1:]
    train_ds = Dataset.from_list(train)
    val_ds = Dataset.from_list(val)
    ds = DatasetDict({'train': train_ds, 'validation': val_ds})
    print(ds)
    return ds

if __name__ == '__main__':
    ds = build_dataset()
    # Save locally for inspection (optional)
    out_dir = os.path.join(os.path.dirname(BASE), 'data', 'hf_dataset')
    os.makedirs(out_dir, exist_ok=True)
    ds.save_to_disk(out_dir)
    print('Saved dataset to', out_dir)
