import argparse
import os
from subprocess import call

def execute_script(script_name, input_directory, output_directory):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_directory, script_name)
    try:
        call(['python', script_path, '--i', input_directory, '--o', output_directory])
    except Exception as e:
        print(f"Error executing script: {e}")

def execute_another_script(script_name, output_directory, results_directory):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_directory, script_name)
    try:
        call(['python', script_path, '--o', output_directory, '--r', results_directory])
    except Exception as e:
        print(f"Error executing script: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Extractor')
    parser.add_argument('--i', type=str, help='Input directory path')
    parser.add_argument('--o', type=str, help='Output directory path')
    parser.add_argument('--r', type=str, help='Results directory path')
    parser.add_argument('--feature_extracter', help='Name of the feature extractor script')
    parser.add_argument('--clustering', help='Name of the clustering script')
    parser.add_argument('--mlp', help='Name of the MLP script')
    args = parser.parse_args()

    if not (args.i and args.o and args.r):
        print("Error: Please provide both the input directory and output directory.")
        exit(1)

    if args.feature_extracter:
        execute_script(args.feature_extracter, args.i, args.o)
    else:
        execute_script('feature_extracter.py', args.i, args.o)

    if args.clustering:
        execute_another_script(args.clustering, args.o, args.r)
    else:
        execute_another_script('clustering.py', args.o, args.r)

    if args.mlp:
        execute_another_script(args.mlp, args.o, args.r)
    else:
        execute_another_script('mlp.py', args.o, args.r)