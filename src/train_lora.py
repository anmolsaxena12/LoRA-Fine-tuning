import os
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
from prepare_data import build_dataset

# Config - edit if needed
MODEL_NAME = os.environ.get('BASE_MODEL', 'distilbert-base-uncased')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', '../checkpoints/lora-distilbert')

def preprocess_function(examples, tokenizer):
    # For QA, we keep question and context; real pipelines need special token handling
    tokenized = tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length', max_length=256)
    # The trainer expects labels for QA models; for simplicity, we don't compute start/end here.
    return tokenized

def main():
    # Build or load dataset
    ds = build_dataset()  # small dataset from src/prepare_data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = ds.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    train_ds = tokenized['train']
    val_ds = tokenized['validation']

    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    # Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.QUESTION_ANSWERING,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy='no',
        save_strategy='epoch',
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_dir='../logs',
        fp16=torch.cuda.is_available(),
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    print('Saved LoRA-tuned model to', OUTPUT_DIR)

if __name__ == '__main__':
    main()
