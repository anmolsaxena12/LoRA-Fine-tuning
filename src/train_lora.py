import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from prepare_data import build_dataset, MODEL_NAME
import evaluate

# Load dataset (now tokenized with start_positions & end_positions)
dataset = build_dataset()
train_ds = dataset["train"].shuffle(seed=42)   # shuffle for randomness
val_ds = dataset["validation"]

# Load tokenizer (so Trainer can decode inputs for metrics/logging)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load base model
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# Apply LoRA adapters
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"],  # LoRA applied to DistilBERT attention layers
    bias="none"
)
model = get_peft_model(model, peft_config)


# Load SQuAD metric from HF evaluate
squad_metric = evaluate.load("squad")

import collections
import numpy as np
import evaluate

squad_metric = evaluate.load("squad")

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size=20, max_answer_length=30):
    # raw_predictions = (start_logits, end_logits)
    start_logits, end_logits = raw_predictions
    predictions = collections.OrderedDict()

    for i, feature in enumerate(features):
        example_id = feature["example_id"]
        context = feature["context"]
        start_logit = start_logits[i]
        end_logit = end_logits[i]

        # get n-best start/end pairs
        start_indexes = np.argsort(start_logit)[-n_best_size:][::-1]
        end_indexes = np.argsort(end_logit)[-n_best_size:][::-1]

        valid_answers = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                if start_index <= end_index and end_index - start_index + 1 <= max_answer_length:
                    answer_text = context[start_index:end_index+1]
                    score = start_logit[start_index] + end_logit[end_index]
                    valid_answers.append({"text": answer_text, "score": score})

        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x["score"])
            predictions[example_id] = best_answer["text"]
        else:
            predictions[example_id] = ""
    return predictions


def compute_metrics(p):
    # p.predictions = (start_logits, end_logits)
    predictions = postprocess_qa_predictions(
        examples=val_ds,
        features=val_ds,
        raw_predictions=p.predictions
    )
    formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in val_ds]
    return squad_metric.compute(predictions=formatted_predictions, references=references)


# ✅ Custom Trainer with correct compute_loss for QA tasks
class QATrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Hugging Face Trainer expects 'labels', but QA uses start/end positions
        start_positions = inputs.pop("start_positions")
        end_positions = inputs.pop("end_positions")
        outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Training arguments
args = TrainingArguments(
    output_dir="../checkpoints/lora-distilbert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_ratio=0.1,
    num_train_epochs=20,
    weight_decay=0.00,
    logging_dir="../logs",
    fp16=torch.cuda.is_available()  # use mixed precision if GPU available
)

# Trainer
trainer = QATrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("../checkpoints/lora-distilbert/final")
    print("✅ LoRA fine-tuned model saved!")
