import glob
import json
import pandas as pd

def remove_newlines(input_string):
    return input_string.replace('\n', ' ').replace('\r', ' ').strip()

def load_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return remove_newlines(content)

def process_txt_files(file_paths):
    txt_data = []
    for file_path in file_paths:
        data = load_from_file(file_path)
        txt_data.append(data)
    return txt_data

def process_json_files(file_paths):
    json_data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
        json_data.append(content['authors'])
    return json_data

file_path = r'C:\Users\Eddie\Downloads\pan22\dataset2\train'

txt_files = glob.glob(file_path + r'\*.txt')
json_files = glob.glob(file_path + r'\*.json')

txt_data = process_txt_files(txt_files)
json_data = process_json_files(json_files)

df = pd.DataFrame({'txt_data': txt_data, 'no_authors': json_data})

# Save DataFrame as CSV
df.to_csv('data.csv', index=False)

print(df.head())