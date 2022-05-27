import argparse
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run few shot experiments.")
    parser.add_argument("-c", "config_filepath", type=str, required=True, help="")

    args = parser.parse_args()
