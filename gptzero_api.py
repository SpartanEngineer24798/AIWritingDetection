import argparse
from gptzero import GPTZeroAPI
import json
import matplotlib.pyplot as plt
import os

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

def make_predictions(api_key, data, output_dir, counter_limit=None):
    gptzero_api = GPTZeroAPI(api_key)
    predictions = {}

    if counter_limit is not None:
        counter = 0

    for elem in data:
        if counter_limit is not None and counter >= counter_limit:
            break

        prediction = gptzero_api.text_predict(data[elem][0])
        predictions[elem] = prediction

        with open(f"{output_dir}/gptzero_response_{elem}.txt", "w") as txt_file:
            txt_file.write(json.dumps(prediction))

        if counter_limit is not None:
            counter += 1

    return predictions
def extract_probs(predictions):
    probs_only = []
    for n in range(1, len(predictions) + 1):
        probs_only.append(predictions[str(n)]['documents'][0]['completely_generated_prob'])
    return probs_only

def plot_histogram(ai_probs_only, human_probs_only):
    n_bins = 20
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    axs.hist(ai_probs_only, bins=n_bins, color='blue', alpha=0.5, label='AI')
    axs.hist(human_probs_only, bins=n_bins, color='red', alpha=0.5, label='Human')

    plt.title("GPTZero Predictions on AI vs. Human Text")
    plt.xlabel("Probability that given text is AI-generated")
    plt.ylabel("No. samples per bin")

    axs.legend(loc='upper right')

    plt.show()

def make_requests(input_dir: str, output_dir: str, api_key: str, counter_limit: int):
    data_ai, data_human = load_dataset(input_dir)

    predictions_ai = make_predictions(api_key, data_ai, output_dir, counter_limit)
    predictions_human = make_predictions(api_key, data_human, output_dir, counter_limit)

    ai_probs_only = extract_probs(predictions_ai)
    human_probs_only = extract_probs(predictions_human)

    plot_histogram(ai_probs_only, human_probs_only)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON data and generate GPTZero predictions.")
    parser.add_argument("--input_dir", required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory for saving processed data.")
    parser.add_argument("--api_key", required=True, help="GPTZero API key for making predictions.")
    parser.add_argument("--counter_limit", type=int, help="Limit the number of API requests made.")
    args = parser.parse_args()

    make_requests(args.input_dir, args.output_dir, args.api_key, args.counter_limit)