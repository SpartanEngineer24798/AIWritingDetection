import os
import pandas as pd
import argparse

def main(csv_file_path):
    output_folder = 'input'
    ai_folder = os.path.join(output_folder, 'ai')
    human_folder = os.path.join(output_folder, 'human')

    # Create the output folders if they don't exist
    os.makedirs(ai_folder, exist_ok=True)
    os.makedirs(human_folder, exist_ok=True)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    for index, row in df.iterrows():
        text = row['txt']
        polarity = row['polarity']

        if polarity == 1:
            target_folder = ai_folder
        else:
            target_folder = human_folder

        file_path = os.path.join(target_folder, f"text{index + 1}.txt")

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process CSV file and split data into AI and human folders.")
    parser.add_argument("--path", type=str, help="Path to the CSV file")
    args = parser.parse_args()
    main(args.path)
