import json
import argparse
from datasets import load_dataset
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--split", type=str, default="validation")
    args = parser.parse_args()
    data = [json.loads(line) for line in open(args.data)]
    qasper_dataset = load_dataset("qasper", split=args.split)
    question_first_word_dict = {}
    first_word_counts = defaultdict(int)
    for qasper_datum in qasper_dataset:
        for question_id, question in zip(qasper_datum['qas']['question_id'], qasper_datum['qas']['question']):
            first_word = question.lower().split()[0]
            first_word_counts[first_word] += 1
            question_first_word_dict[question_id] = first_word
    lexical_features = [word for word, count in first_word_counts.items() if count >= 10]
    for datum in data:
        question_first_word = question_first_word_dict[datum['id']]
        for feature in lexical_features:
            if feature == question_first_word:
                datum[feature] = 1.0
            else:
                datum[feature] = 0.0

    with open(args.data, "w") as outfile:
        for datum in data:
            print(json.dumps(datum), file=outfile)


if __name__ == "__main__":
    main()
