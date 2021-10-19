import random
import json
import argparse
from datasets import load_dataset


random.seed(20391)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--squad_dev", type=str, required=True)
    parser.add_argument("--eqqa_train_output", type=str, required=True)
    parser.add_argument("--eqqa_dev_output", type=str, required=True)
    args = parser.parse_args()
    dev_set_with_f1s = json.load(open(args.squad_dev))
    f1s = {d["qid"]: d["max_f1"] for d in dev_set_with_f1s}
    squad_dev = load_dataset("squad_v2", split="validation")
    eqqa_data = []
    for datum in squad_dev:
        eqqa_data.append({
            "id": datum["id"],
            "context": datum["context"],
            "question": datum["question"],
            "max_f1": f1s[datum["id"]]
        })

    random.shuffle(eqqa_data)
    num_train = int(len(eqqa_data) * 0.9)
    train_set = eqqa_data[:num_train]
    dev_set = eqqa_data[num_train:]

    with open(args.eqqa_train_output, "w") as outfile:
        for datum in train_set:
            print(json.dumps(datum), file=outfile)

    with open(args.eqqa_dev_output, "w") as outfile:
        for datum in dev_set:
            print(json.dumps(datum), file=outfile)


if  __name__ == "__main__":
    main()
