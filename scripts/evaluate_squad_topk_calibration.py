import json
import argparse
from tqdm import tqdm
from allennlp_models.rc.tools import squad

f1_hash = {}
def compute_f1(pred, target) -> float:
    pred = pred.strip()
    target = target.strip()
    if (pred, target) not in f1_hash:
        f1_hash[(pred, target)] = squad.compute_f1(pred, target)
    return f1_hash[(pred, target)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    oracle_f1s = []
    top_f1s = []
    oracle_f1_ranks = []
    num_flips = []
    for line in tqdm(open(args.predictions)):
        data = json.loads(line)
        for i in range(len(data["id"])):
            answers = data["answers"][i]
            predictions = data["best_span_str"][i][:args.k]
            if not answers:
                f1s = [1.0 if not prediction else 0.0 for prediction in predictions]
            else:
                f1s = [max([compute_f1(prediction, answer) for answer in answers])
                       for prediction in predictions]

            datum_num_flips = 0
            for j in range(len(f1s) - 1):
                for k in range(j+1, len(f1s)):
                    if f1s[j] < f1s[k]:
                        print(predictions[j], f1s[j], predictions[k], f1s[k])
                        datum_num_flips += 1

            num_flips.append(datum_num_flips)
            top_f1s.append(f1s[0])
            oracle_f1 = max(f1s)
            oracle_f1s.append(oracle_f1)
            for i, f1 in enumerate(f1s):
                if oracle_f1 == f1:
                    break
            oracle_f1_ranks.append(i + 1)

    mean = lambda x: sum(x) / len(x)
    print(f"Top F1: {mean(top_f1s)}")
    print(f"Oracle F1: {mean(oracle_f1s)}")
    print(f"Oracle F1 rank: {mean(oracle_f1_ranks)}")
    print(f"Num F1 flips: {mean(num_flips)}")

if __name__ == "__main__":
    main()
