# LoRA Fine-Tuning Project (proj-2-lora-finetune)

This project demonstrates a small end-to-end LoRA (PEFT) fine-tuning workflow for a question-answering task.
It's intentionally lightweight and meant for learning / portfolio purposes.

## Contents
- `data/support_tickets.json` - small sample dataset in SQuAD-like format
- `src/prepare_data.py` - convert sample JSON -> HF Dataset format (and save locally)
- `src/train_lora.py` - train a LoRA adapter on top of a base model (DistilBERT by default)
- `src/inference.py` - run a quick QA using the saved LoRA checkpoint
- `src/evaluate.py` - simple EM + F1 evaluator (expects preds.json and refs.json in data/)
- `requirements.txt` - Python dependencies

## Quick start (local)
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Prepare dataset:
   ```bash
   python src/prepare_data.py
   ```

3. Train LoRA adapter (small debug run):
   ```bash
   python src/train_lora.py
   ```
   - By default this uses `distilbert-base-uncased`. To change, set env var `BASE_MODEL`.
   - The script saves the adapter / model to `../checkpoints/lora-distilbert` by default.

4. Inference (after training/saving):
   ```bash
   python src/inference.py
   ```

5. Evaluate:
   - Use `src/inference.py` to save predictions to `data/preds.json` and ground-truth to `data/refs.json` then run:
   ```bash
   python src/evaluate.py
   ```

## Notes & next steps
- This example is intentionally small: increase data, epochs, and batch size for real experiments.
- For production, use `accelerate` for multi-GPU, proper QA token handling (start/end positions), and better evaluation harness.
- Consider using `bitsandbytes` for 8/4-bit quantized inference to reduce GPU memory usage.
