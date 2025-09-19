import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from prepare_data import build_dataset, MODEL_NAME
import evaluate

# Load dataset (input_text + labels already prepared)
dataset = build_dataset()
train_ds = dataset["train"].shuffle(seed=42)
val_ds = dataset["validation"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load GPT-2 model for causal LM
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))  # important if pad token was added

# Apply LoRA adapters for causal LM
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn"],  # common choice for GPT-2 attention layer
    bias="none"
)
model = get_peft_model(model, peft_config)

# Simple compute_metrics: generate text & compare to reference
squad_metric = evaluate.load("squad")
def compute_metrics(eval_pred):
    preds = []
    refs = []
    model.eval()
    for ex in val_ds:
        # Tokenize input text
        inputs = tokenizer(ex["input_text"], return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs.input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,  # ✅ named argument
                max_new_tokens=64,
                pad_token_id=tokenizer.eos_token_id
            )

        pred_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        preds.append({"id": ex["id"], "prediction_text": pred_text.strip()})

        # ✅ Add dummy answer_start for SQuAD metric
        refs.append({"id": ex["id"], "answers": [{"text": ex["target_text"], "answer_start": 0}]})

    return squad_metric.compute(predictions=preds, references=refs)

# Training arguments
args = TrainingArguments(
    output_dir="../checkpoints/lora-gpt2",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",  # use F1 score to select best model
    greater_is_better=True,
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=20,
    logging_dir="../logs",
    fp16=torch.cuda.is_available()
)

# Trainer (no need for custom class — Trainer handles loss automatically)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("../checkpoints/lora-gpt2/final")
    print("✅ LoRA fine-tuned GPT-2 model saved!")
