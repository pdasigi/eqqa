import argparse
import json
import math
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--type", type=str, default="qasper")
    parser.add_argument("--quantiles", type=int, default=3)
    args = parser.parse_args()
    f1s = []
    if args.type == "qasper":
        data = json.load(open(args.data))

        for datum in data.values():
            for qa_info in datum["qas"]:
                f1s.append(qa_info["max_f1"])
    else:
        f1s = [json.loads(line)["pred_max_f1"] for line in open(args.data)]

    max_points_per_quantile = len(f1s) // args.quantiles
    boundaries = []
    points_in_quantile = []
    for each_f1 in sorted(f1s):
        points_in_quantile.append(each_f1)
        if len(points_in_quantile) >= max_points_per_quantile:
            boundaries.append((each_f1, len(points_in_quantile)))
            points_in_quantile = []
    if len(boundaries) < args.quantiles:
        boundaries.append((each_f1, len(points_in_quantile)))

    print(f"Quantile boundaries: {boundaries}")
    plt.hist(f1s, 50)
    plt.show()


if __name__ == "__main__":
    main()
