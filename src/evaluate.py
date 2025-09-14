import os
import json
import re
from collections import Counter

def normalize_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = ' '.join(s.split())
    return s

def exact_match(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))

def f1_score(prediction, ground_truth):
    p_tokens = normalize_text(prediction).split()
    g_tokens = normalize_text(ground_truth).split()
    common = Counter(p_tokens) & Counter(g_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(p_tokens)
    recall = num_same / len(g_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate(preds, refs):
    em = sum(exact_match(p, r) for p, r in zip(preds, refs)) / len(preds)
    f1 = sum(f1_score(p, r) for p, r in zip(preds, refs)) / len(preds)
    return {'exact_match': em, 'f1': f1}

if __name__ == '__main__':
    # Example usage: use evaluate after running inference.py and saving predictions
    preds_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'preds.json')
    refs_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'refs.json')
    if not os.path.exists(preds_file) or not os.path.exists(refs_file):
        print('Run inference first to produce preds.json and refs.json in data/')
    else:
        preds = json.load(open(preds_file))
        refs = json.load(open(refs_file))
        # preds and refs expected to be lists of strings of same length
        metrics = evaluate(preds, refs)
        print('Evaluation:', metrics)
