import os
from transformers import pipeline

MODEL_DIR = os.environ.get('MODEL_DIR', '../checkpoints/lora-distilbert/final')
# If you used the train script above, output dir may differ; point MODEL_DIR accordingly.

def run_sample():
    if not os.path.exists(MODEL_DIR):
        print('Model directory not found:', MODEL_DIR)
        return
    qa = pipeline('question-answering', model=MODEL_DIR, tokenizer=MODEL_DIR)
    question = 'What error does the user see when uploading large CSV files?'
    context = 'User reports that the application throws a 500 error when uploading large CSV files. The error occurs after a timeout and the user sees Internal Server Error.'
    out = qa(question=question, context=context)
    print(out)

if __name__ == '__main__':
    run_sample()
