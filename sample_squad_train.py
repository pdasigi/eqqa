import random
import math
import json
import argparse
from datasets import load_dataset

random.seed(4029)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--percentage', type=float, help='Percentage of dataset to output', default='0.5')
    parser.add_argument('--output', type=str, help='Location where dataset will be written', default='data/squad_train_0.5.json')
    args = parser.parse_args()
    squad_train = list(load_dataset('squad_v2', split='train'))
    random.shuffle(squad_train)
    max_num_instances = math.ceil(len(squad_train) * args.percentage)
    num_instances = 0
    output_instances = []
    for instance in squad_train:
        output_instances.append(
                {
                    "title": instance["title"],
                    "context": instance["context"],
                    "qas": [
                        {
                            "question": instance["question"],
                            "answers": [{"text": t, "answer_start": s} for t, s in zip(instance["answers"]["text"], instance["answers"]["answer_start"])],
                            "id": instance["id"]
                        }
                    ]
                }
        )
        num_instances += 1
        if num_instances >= max_num_instances:
            break

    data = {"data": [{"paragraphs": output_instances}]}
    print(f"Sampled {max_num_instances} instances ({args.percentage}), and writing to {args.output}")
    with open(args.output, "w") as outfile:
        json.dump(data, outfile, indent=2)

