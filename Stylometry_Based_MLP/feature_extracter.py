import json
import re
import numpy as np
import string

import statistics
from lexicalrichness import LexicalRichness
import readability
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import torch
from spacy import util
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import os
import sys
import subprocess
import argparse
from collections import defaultdict
from tqdm import tqdm

def execute_another_script():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    another_script = os.path.join(current_directory, 'another_script.py')
    subprocess.call(['python', another_script])

def word_count(document):
    tokens = word_tokenize(document)
    nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    filtered = [w for w in tokens if nonPunct.match(w)]
    return len(filtered)

def word_count_sent(document):
    tokens = sent_tokenize(document)
    nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    filtered = [w for w in tokens if nonPunct.match(w)]
    word_counts = [word_count(sent) for sent in filtered]
    if len(word_counts) == 0:
        return 0, 0
    mean = sum(word_counts) / len(word_counts)
    if len(word_counts) < 2:
        stdev = 0
    else:
        stdev = statistics.stdev(word_counts)
    return mean, stdev

def special_punc_count_sent(document):
    special_puncts = ['!', '\"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']
    tokens = sent_tokenize(document)
    nonPunct = re.compile('.*[A-Za-z0-9].*')  # must contain a letter or digit
    filtered = [w for w in tokens if nonPunct.match(w)]
    punct_count = 0
    total_sentences = len(filtered)
    if total_sentences == 0:
        return 0
    for sent in filtered:
        for char in sent:
            if char in special_puncts:
                punct_count += 1
    return float(punct_count) / total_sentences


def readability_score(document):
    try:
        r = readability.getmeasures(document, lang='en')
        fk = r['readability grades']['Kincaid']
        f = r['readability grades']['FleschReadingEase']
        ari = r['readability grades']['ARI']
    except:
        return 0, 0, 0
    else:
        return fk, f, ari


def lexical_richness(document):
    sample_size = 10
    iterations = 50
    lex = LexicalRichness(document)
    ret_list = []
    words = document.split()
    try:
        if len(words) > 45:
            ret_list.append(lex.mattr(window_size=25))
        else:
            window_size = max(1, len(words) // 3)  # Adjusted window size
            if window_size > len(words):
                window_size = len(words)
            ret_list.append(lex.mattr(window_size=window_size))
    except Exception:
        ret_list.append(0)  # Return 0 if an exception is thrown during feature extraction
    ret_list.append(lex.mtld(threshold=0.72))
    return ret_list


class PerplexityCalculator:
    STRIDE = 512
    MODEL = "gpt2"
    SPACY_MODEL = "en_core_web_trf"
    NLP_MAX_LENGTH = 2000000

    def __init__(self):
        self._tokenizer = GPT2Tokenizer.from_pretrained(self.MODEL)
        self._model = GPT2LMHeadModel.from_pretrained(self.MODEL)
        self._max_length = self._model.config.n_positions
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model.to(self._device)
        self._nlp = spacy.load(self.SPACY_MODEL)
        self._nlp.add_pipe("sentencizer")

        infixes = self._nlp.Defaults.infixes + ['`', "'", '"']
        self._nlp.tokenizer.infix_finditer = util.compile_infix_regex(infixes).finditer
        self._count_vectorizer_dict = {}
        self._tfidf_transformer_dict = {}
        self._tfidf_vectors = {}

    def calculate_inner(self, in_text: str, precision=6) -> float:
        encodings = self._tokenizer(in_text, return_tensors='pt', max_length=self.NLP_MAX_LENGTH, truncation=True)
        seq_len = encodings.data['input_ids'].size(1)

        nlls = []
        prev_end_loc = 0
        count = 0
        for begin_loc in range(0, seq_len, self.STRIDE):
            end_loc = min(begin_loc + self._max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self._device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self._model(input_ids, labels=target_ids)
                nlls.append(outputs.loss)

            prev_end_loc = end_loc
            count += 1
            if end_loc == seq_len:
                break
        return round(torch.exp(torch.stack(nlls).mean()).item(), precision)


def ppl_calculator(text: str, precision=2) -> float:
    ppl_calc = PerplexityCalculator()
    return ppl_calc.calculate_inner(text, precision)


def extract_features(document):
    results = []

    words_per_sent = word_count_sent(document)
    results.append(words_per_sent[0])
    results.append(words_per_sent[1])

    special_punc_sent_result = special_punc_count_sent(document)
    results.append(special_punc_sent_result)

    readability_results = readability_score(document)
    results.extend(readability_results)

    lexical_richness_results = lexical_richness(document)
    results.extend(lexical_richness_results)

    results.append(ppl_calculator(document))

    return results

def process_data(input_directory, output_directory):
    extract = defaultdict(list)

    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' does not exist.")
        return

    total_iterations = 0
    for key in os.listdir(input_directory):
        subfolder = os.path.join(input_directory, key)
        if not os.path.isdir(subfolder):
            print(f"Error: '{subfolder}' is not a valid directory.")
            continue
        total_iterations += len(os.listdir(subfolder))

    progress_bar = tqdm(total=total_iterations, desc='Processing')

    for key in os.listdir(input_directory):
        subfolder = os.path.join(input_directory, key)
        if not os.path.isdir(subfolder):
            print(f"Error: '{subfolder}' is not a valid directory.")
            continue

        output_subfolder = os.path.join(output_directory, key)
        os.makedirs(output_subfolder, exist_ok=True)

        for filename in os.listdir(subfolder):
            file_path = os.path.join(subfolder, filename)
            if not os.path.isfile(file_path):
                print(f"Error: '{file_path}' is not a valid file.")
                continue

            with open(file_path, 'r', encoding='utf-8') as file:
                string = file.read()

            extracted_features = extract_features(string)
            extract[key].append(extracted_features)

            output_filename = os.path.splitext(filename)[0] + '.json'
            output_path = os.path.join(output_subfolder, output_filename)
            with open(output_path, 'w') as output_file:
                json.dump(extracted_features, output_file)

            progress_bar.update(1)

    progress_bar.close()


def execute_another_script(script_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_directory, script_name)
    subprocess.call(['python', script_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Extractor')
    parser.add_argument('--i', type=str, help='Input directory')
    parser.add_argument('--o', type=str, help='Output directory')
    parser.add_argument('--clustering', help='Name of clustering script')
    parser.add_argument('--mlp', help='Name of MLP script')
    args = parser.parse_args()

    if args.i and args.o:
        process_data(args.i, args.o)
        input_folder = args.o
    else:
        print("Error: Please provide both the input directory and output directory.")
        exit(1)

    if args.clustering:
        execute_another_script(args.clustering, input_folder)
    else:
        execute_another_script('clustering.py', input_folder)

    if args.mlp:
        execute_another_script(args.mlp, input_folder)
    else:
        execute_another_script('mlp.py', input_folder)