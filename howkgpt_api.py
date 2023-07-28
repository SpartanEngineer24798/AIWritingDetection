import argparse
import requests
import json
import spacy
import torch
from spacy import util
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import os
from tqdm import tqdm

class TextObj:
    def __init__(self, the_text: str):
        self.the_text = the_text

BASE_URL = 'https://howkgpt-f02f.uc.r.appspot.com'
TOKEN_URL = f"{BASE_URL}/api/token"
PPL_URL = f"{BASE_URL}/api/perplexity"

def get_token(identity: str) -> str:
    data = '{"identity": "' + identity + '"}'
    response = requests.put(TOKEN_URL, headers={'content-type': 'application/json'}, data=data)
    return response.json()['bearer']

def get_ppl(req: TextObj, bearer_token: str) -> str:
    headers = {
        'authorization': f'Bearer {bearer_token}',
        'content-type': 'application/json'
    }
    response = requests.put(PPL_URL, headers=headers, data=json.dumps(req.__dict__))
    return response.text

def load_dataset(input_dir: str):
    data_ai = {}
    data_human = {}
    ai_dir = os.path.join(input_dir, "ai")
    human_dir = os.path.join(input_dir, "human")

    for file_name in os.listdir(ai_dir):
        with open(os.path.join(ai_dir, file_name), "r", encoding="utf-8", errors="ignore") as read_file:
            data_ai[file_name] = [read_file.read()]

    for file_name in os.listdir(human_dir):
        with open(os.path.join(human_dir, file_name), "r", encoding="utf-8", errors="ignore") as read_file:
            data_human[file_name] = [read_file.read()]

    return data_ai, data_human

def make_requests(input_dir: str, output_dir: str, api_key, counter_limit=None):
    data_ai, data_human = load_dataset(input_dir)

    total_requests = len(data_ai) + len(data_human)

    true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0

    counter = 0
    for idx, (file_name, text) in enumerate(data_ai.items(), start=1):
        counter += 1
        req = TextObj(text[0])
        bearer_token = get_token(api_key)
        api_response = get_ppl(req, bearer_token)
        result = json.loads(api_response)["result"]

        if result == "AI":
            true_positive += 1
        else:
            false_positive += 1

        with open(os.path.join(output_dir, f"response_ai_{idx}.json"), "w") as f:
            f.write(api_response)

        if counter_limit is not None and counter >= counter_limit:
            break

    counter = 0
    for idx, (file_name, text) in enumerate(data_human.items(), start=1):
        counter += 1
        req = TextObj(text[0])
        bearer_token = get_token(api_key)
        api_response = get_ppl(req, bearer_token)
        result = json.loads(api_response)["result"]

        if result == "Human":
            true_negative += 1
        else:
            false_negative += 1

        with open(os.path.join(output_dir, f"response_human_{idx}.json"), "w") as f:
            f.write(api_response)

        if counter_limit is not None and counter >= counter_limit:
            break

    return true_positive, false_positive, true_negative, false_negative

def plot_confusion_matrix(true_positive, false_positive, true_negative, false_negative):
    confusion_matrix = np.array([[true_negative, false_positive],
                                 [false_negative, true_positive]])

    classes = ['Negative', 'Positive']

    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    for i in range(2):
        for j in range(2):
            text = plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')
            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process API requests.")
    parser.add_argument("--i", dest="input_dir", required=True, help="Input directory path.")
    parser.add_argument("--o", dest="output_dir", required=True, help="Output directory path.")
    parser.add_argument("--api", dest="api_key", required=True, help="API key.")
    parser.add_argument("--debug_plot", action="store_true", help="Use debug values for confusion matrix.")
    parser.add_argument("--counter_limit", type=int, help="Limit the number of requests based on the counter value.")
    args = parser.parse_args()

    if args.debug_plot:
        true_positive = 994
        false_positive = 810
        true_negative = 23
        false_negative = 4
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        true_positive, false_positive, true_negative, false_negative = make_requests(
            args.input_dir, args.output_dir, api_key=args.api_key, counter_limit=args.counter_limit
        )

    plot_confusion_matrix(true_positive, false_positive, true_negative, false_negative)