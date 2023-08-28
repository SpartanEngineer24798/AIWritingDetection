import argparse
import os
import subprocess

def execute_script(script_name, input_directory, output_directory):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_directory, script_name)

    python_versions = ['python3', 'python']

    for python_version in python_versions:
        try:
            subprocess.call([python_version, script_path, '--i', input_directory, '--o', output_directory])
            break  # If successful, exit the loop
        except Exception as e:
            print(f"Error executing script with {python_version}: {e}")

def execute_another_script(script_name, output_directory, results_directory):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_directory, script_name)

    python_versions = ['python3', 'python']

    for python_version in python_versions:
        try:
            subprocess.call([python_version, script_path, '--o', output_directory, '--r', results_directory])
            break  # If successful, exit the loop
        except Exception as e:
            print(f"Error executing script with {python_version}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature Extractor')
    parser.add_argument('--i', type=str, help='Input directory path', required=True)
    parser.add_argument('--o', type=str, help='Output directory path')
    parser.add_argument('--r', type=str, help='Results directory path')
    parser.add_argument('--feature_extracter', help='Name of the feature extractor script')
    parser.add_argument('--clustering', help='Name of the clustering script')
    parser.add_argument('--mlp', help='Name of the MLP script')
    parser.add_argument('--only_extraction', action='store_true', help='Run only the feature extraction script')
    parser.add_argument('--only_clustering', action='store_true', help='Run only the clustering script')
    parser.add_argument('--only_mlp', action='store_true', help='Run only the MLP script')
    args = parser.parse_args()

    # Check if output and results directories are provided, otherwise create folders "output" and "results"
    current_directory = os.path.abspath('.')
    output_directory = args.o if args.o else os.path.join(current_directory, "output")
    results_directory = args.r if args.r else os.path.join(current_directory, "results")

    if not os.path.exists(args.i):
        print("Error: Input directory does not exist.")
        exit(1)

    if os.path.exists(output_directory) and not args.o:
        print("Warning: Output directory already exists. Specify a different output directory or use '--o' to overwrite.")
        exit(1)

    if os.path.exists(results_directory) and not args.r:
        print("Warning: Results directory already exists. Specify a different results directory or use '--r' to overwrite.")
        exit(1)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    print("Beginning main script.")

    if args.only_extraction:
        if args.feature_extracter:
            execute_script(args.feature_extracter, args.i, output_directory)
        else:
            execute_script('feature_extracter.py', args.i, output_directory)
    elif args.only_clustering:
        if args.clustering:
            execute_another_script(args.clustering, output_directory, results_directory)
        else:
            execute_another_script('clustering.py', output_directory, results_directory)
    elif args.only_mlp:
        if args.mlp:
            execute_another_script(args.mlp, output_directory, results_directory)
        else:
            execute_another_script('mlp.py', output_directory, results_directory)
    else:
        if args.feature_extracter:
            execute_script(args.feature_extracter, args.i, output_directory)
        else:
            execute_script('feature_extracter.py', args.i, output_directory)

        if args.clustering:
            execute_another_script(args.clustering, output_directory, results_directory)
        else:
            execute_another_script('clustering.py', output_directory, results_directory)

        if args.mlp:
            execute_another_script(args.mlp, output_directory, results_directory)
        else:
            execute_another_script('mlp.py', output_directory, results_directory)
