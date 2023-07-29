import os
import openai
import logging
import json
import datetime
import time
from tqdm import tqdm
import argparse

def read_json_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def read_personalities_from_file(file_path):
    with open(file_path, 'r') as file:
        personalities = [line.strip() for line in file]
    return personalities

def extract_prompts(json_data):
    prompts = []
    for result in json_data:
        if result['source'] == "reddit_eli5":
            continue
        prompts.append(result['question'])
    return prompts

def save_log_file(request, response, file_path):

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_name = f"{file_path}/log_file_{timestamp}.txt"

    if os.path.isfile(file_name):
        with open(file_name, 'r') as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                pass

    log_entry = {
        'timestamp': str(datetime.datetime.now()),
        'request': request,
        'response': response
    }

    with open(file_name, 'w') as file:
        json.dump(log_entry, file, indent=4)

def create_personality_folder(personality, root_path):
    folder_path = os.path.join(root_path, personality)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def generate_personality_data(personalities, prompts, api_log_root, api_key, counter_limit, model):
    openai.api_key = api_key

    for personality_index, personality in enumerate(personalities):
        personality_folder = create_personality_folder(personality, api_log_root)
        quit_program = False

        for request_index in tqdm(range(0, min(counter_limit, len(prompts))), desc=f"Personality {personality_index+1}/{len(personalities)}"):
            text = prompts[request_index] + " Please answer in less than 200 words. Please respond like " + personality

            request = [{"role": "user", "content": text}]

            retry_count = 0
            invalid_request_error = False

            while retry_count < 5:
                try:
                    response = openai.ChatCompletion.create(
                    model=str(model),
                    messages=request
                    )
                    save_log_file(request, response, personality_folder)
                    break

                except openai.error.InvalidRequestError as e:
                    save_log_file(request, {'error': str(e)}, personality_folder)
                    print(f"Invalid request error occurred: {e}")
                    invalid_request_error = True
                    break

                except openai.error.APIConnectionError as e:
                    save_log_file(request, {'error': str(e)}, personality_folder)
                    print("API Connection Error occurred. Quitting...")
                    quit_program = True
                    break

                except openai.error.AuthenticationError as e:
                    save_log_file(request, {'error': str(e), 'info': 'Retrying...'}, personality_folder)
                    print("Authentication Error occurred. Quitting...")
                    quit_program = True
                    break

                except (openai.error.Timeout) as e:
                    print("Waiting for 1 minute before retrying...")
                    save_log_file(request, {'error': str(e), 'info': 'Retrying...'}, personality_folder)
                    time.sleep(60)
                    retry_count += 1

                except (openai.error.RateLimitError) as e:
                    if str(e) == "You exceeded your current quota, please check your plan and billing details.":
                        print("\nError! Quota exceeded. Please check your OpenAI dashboard for API quota.")
                        quit_program = True
                        break
                    else:
                        print("Waiting for 1 minute before retrying...")
                        save_log_file(request, {'error': str(e), 'info': 'Retrying...'}, personality_folder)
                        time.sleep(60)
                        retry_count += 1

                except (openai.error.ServiceUnavailableError) as e:
                    print("Waiting for 1 minute before retrying...")
                    save_log_file(request, {'error': str(e), 'info': 'Retrying...'}, personality_folder)
                    time.sleep(60)
                    retry_count += 1

                except (openai.error.APIError) as e:
                    print("Waiting for 1 minute before retrying...")
                    save_log_file(request, {'error': str(e), 'info': 'Retrying...'}, personality_folder)
                    time.sleep(60)
                    retry_count += 1

            if invalid_request_error:
                continue

            if retry_count >= 5:
                save_log_file(request, {'error': 'Retry limit reached.'}, personality_folder)
                print("Retry limit reached. Retrying after 5 minutes.")
                time.sleep(300)
                break

            if quit_program:
                break

        if quit_program:
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate personality data using OpenAI API.")
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON file.")
    parser.add_argument("--api_logs", type=str, required=True, help="Root directory for saving API logs.")
    parser.add_argument("--key", type=str, required=True, help="Your OpenAI API key.")
    parser.add_argument("--counter_limit", type=int, default=10000, help="Number of requests to be made for each personality.")
    parser.add_argument("--personalities_file", type=str, default="./OpenAI_Data/personalities.txt", help="Path to the .txt file containing personalities.")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI API model to use for generating responses. As of August 2023, only gpt-4 or gpt-3.5-turbo is supported.")
    args = parser.parse_args()

    json_data = read_json_data(args.json)
    prompts = extract_prompts(json_data)
    personalities = read_personalities_from_file(args.personalities_file)

    generate_personality_data(personalities, prompts, args.api_logs, args.key, args.counter_limit, args.model)