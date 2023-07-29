import os
import json
import re
import argparse

def extract_personality_type(request_content):
    match = re.search(r'respond like (.+)', request_content, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    else:
        return 'default'

def extract_response_from_log(log_file):
    with open(log_file, 'r') as file:
        log_data = json.load(file)
        request_content = log_data['request'][0]['content']
        response_string = log_data['response']['choices'][0]['message']['content']
        return request_content, response_string

def save_response_to_txt(response_string, output_file):
    with open(output_file, 'w') as file:
        file.write(response_string)

def process_log_files(log_directory, root_saving_directory):
    for filename in os.listdir(log_directory):
        if filename.endswith('.txt'):
            log_file_path = os.path.join(log_directory, filename)
            request_content, response_string = extract_response_from_log(log_file_path)

            personality_type = extract_personality_type(request_content)

            output_folder = os.path.join(root_saving_directory, personality_type)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            existing_files = os.listdir(output_folder)
            suffix = 1
            while f"{personality_type}{suffix}.txt" in existing_files:
                suffix += 1

            output_filename = f"{personality_type}{suffix}.txt"
            output_file_path = os.path.join(output_folder, output_filename)

            save_response_to_txt(response_string, output_file_path)
            print(f"Response extracted and saved to {output_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and save responses from log files.")
    parser.add_argument("--api_logs", required=True, help="Path to the directory containing log files.")
    parser.add_argument("--personalities_data", required=True, help="Path to the root saving directory.")
    args = parser.parse_args()

    process_log_files(args.api_logs, args.personalities_data)
