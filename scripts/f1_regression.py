import json
import argparse
import random
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error


random.seed(23019)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--features", type=str, nargs="+", required=True)
    parser.add_argument("--target", type=str, default="mean_f1")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    all_data = [json.loads(line) for line in open(args.data)]
    random.shuffle(all_data)
    folds = []
    fold_size = len(all_data) // args.folds
    start_index = 0
    for i in range(args.folds):
        if i == args.folds - 1:
            folds.append(all_data[start_index:])
        else:
            folds.append(all_data[start_index:start_index + fold_size])
            start_index += fold_size

    train_errors = []
    test_errors = []
    for i in range(args.folds):
        test_data = folds[i]
        train_data = []
        for j in range(args.folds):
            if j == i:
                continue
            train_data.extend(folds[j])

        train_x = []
        train_y = []

        for datum in train_data:
            train_x.append([datum[feature] for feature in args.features])
            train_y.append(datum["pred_mean_f1"] if args.target == "mean_f1" else datum["pred_max_f1"])

        test_x = []
        test_y = []

        for datum in test_data:
            test_x.append([datum[feature] for feature in args.features])
            test_y.append(datum["pred_mean_f1"] if args.target == "mean_f1" else datum["pred_max_f1"])

        if args.verbose:
            print(f"Training for fold {i} on {len(train_x)} data points")

        regression_model = linear_model.Ridge(alpha=args.alpha)
        regression_model.fit(train_x, train_y)

        train_predictions = regression_model.predict(train_x)
        train_error = mean_absolute_error(train_y, train_predictions)

        if args.verbose:
            print("Coefficients:", list(zip(args.features, regression_model.coef_)))
            print(f"Train error: {train_error}")

        train_errors.append(train_error)

        test_predictions = regression_model.predict(test_x)
        test_error = mean_absolute_error(test_y, test_predictions)

        if args.verbose:
            print(f"Test error: {test_error}")
        test_errors.append(test_error)

    print(f"Average train error: {np.mean(train_errors)} (+/- {np.std(train_errors)})")
    print(f"Average test error: {np.mean(test_errors)} (+/- {np.std(test_errors)})")

if __name__ == "__main__":
    main()
