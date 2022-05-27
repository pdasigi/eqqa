import argparse
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run few shot experiments.")
    parser.add_argument("-c", "config_filepath", type=str, required=True, help="")
    parser.add_argument("-out", "output_dir", type=str, required=True, help="")
    
    args = parser.parse_args()
    config_filepath = args.config_filepath
    output_dir = args.output_dir

    configs = json.load(open(config_filepath))

    print(f"Results of the experiment will be placed on {output_dir}")
    os.makedirs(output_dir, exist_ok=True)