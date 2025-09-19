import os
from transformers import pipeline

MODEL_DIR = os.environ.get('MODEL_DIR', '../checkpoints/lora-gpt2/checkpoint-10')
# If you used the train script above, output dir may differ; point MODEL_DIR accordingly.

def run_sample():
    if not os.path.exists(MODEL_DIR):
        print('Model directory not found:', MODEL_DIR)
        return
    qa = pipeline('question-answering', model=MODEL_DIR, tokenizer=MODEL_DIR)
    context = 'User reports that the application throws a 500 error when uploading large CSV files. The error occurs after a timeout and the user sees Internal Server Error.'
    question = 'What error does the user see when uploading large CSV files?'
    out = qa(question=question, context=context)
    print(out)

    context = 'User reports that they are not able to message their crush. Also the display picture of crush is not visible.'
    question = 'Why does the user want to contact their crush?'
    out = qa(question=question, context=context)
    print(out)

if __name__ == '__main__':
    run_sample()
