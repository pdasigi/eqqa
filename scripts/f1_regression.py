import json
import argparse
import random
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error


random.seed(23019)


def bucket_accuracy(targets, predictions):
    accuracy = []
    for prediction, target in zip(predictions, targets):
        prediction_bucket = None
        target_bucket = None
        for i, limit in enumerate([0.0, 0.134, 1.0]):
            if prediction <= limit and prediction_bucket is None:
                prediction_bucket = i
            if target <= limit and target_bucket is None:
                target_bucket = i
        accuracy.append(prediction_bucket == target_bucket)
    return sum(accuracy) / len(accuracy)


def get_train_test_errors(train_data, test_data, metric, features, target, alpha, baseline_constant,
                          test_predictions_file, verbose):
    train_x = []
    train_y = []

    for datum in train_data:
        train_x.append([datum[feature] for feature in features])
        train_y.append(datum["pred_mean_f1"] if target == "mean_f1" else datum["pred_max_f1"])

    test_x = []
    test_y = []

    for datum in test_data:
        test_x.append([datum[feature] for feature in features])
        test_y.append(datum["pred_mean_f1"] if target == "mean_f1" else datum["pred_max_f1"])

    if verbose:
        print(f"Training for fold {i} on {len(train_x)} data points")

    regression_model = linear_model.Ridge(alpha=alpha)
    regression_model.fit(train_x, train_y)

    train_predictions = regression_model.predict(train_x)
    train_error = metric(train_y, train_predictions)

    if verbose:
        print("Coefficients:", list(zip(features, regression_model.coef_)))
        print(f"Train error: {train_error}")


    test_predictions = regression_model.predict(test_x)
    test_error = metric(test_y, test_predictions)

    baseline_test_error = {}
    # Measuring errors by directly using the features as predictions
    for i, feature in enumerate(features):
        if feature == "ttd_pairwise_f1_mean":
            if test_predictions_file:
                for target, prediction in zip(test_y, [x[i] for x in test_x]):
                    print(json.dumps({"prediction": prediction, "target": target}), file=test_predictions_file)
        baseline_test_error[feature] = metric(test_y, [x[i] for x in test_x])

    baseline_test_error['constant'] = metric(test_y, [baseline_constant] * len(test_y))

    if verbose:
        print(f"Test error: {test_error}")
        print(f"Baseline test errors: {[baseline_test_errors[feature][-1] for feature in features]}")

    return train_error, test_error, baseline_test_error

def run_cross_validation(data, num_folds, metric, features, target, alpha, baseline_constant,
                         test_predictions_file, verbose):
    random.shuffle(data)
    folds = []
    fold_size = len(data) // num_folds
    start_index = 0
    for i in range(num_folds):
        if i == num_folds - 1:
            folds.append(data[start_index:])
        else:
            folds.append(data[start_index:start_index + fold_size])
            start_index += fold_size
    train_errors = []
    test_errors = []
    baseline_test_errors = {feature: [] for feature in features}
    baseline_test_errors['constant'] = []
    for i in range(num_folds):
        test_data = folds[i]
        train_data = []
        for j in range(num_folds):
            if j == i:
                continue
            train_data.extend(folds[j])
        train_error, test_error, baseline_test_error = get_train_test_errors(train_data, test_data, metric,
                                                                             features, target, alpha,
                                                                             baseline_constant,
                                                                             test_predictions_file, verbose)
        for feature, error in baseline_test_error.items():
            baseline_test_errors[feature].append(error)
        train_errors.append(train_error)
        test_errors.append(test_error)

    return train_errors, test_errors, baseline_test_errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        required=True
    )
    parser.add_argument(
        "--test_data",
        type=str,
        help="If not given, will run CV on 'data'"
    )
    parser.add_argument(
        "--ignore_features",
        type=str,
        nargs="+",
        help="All the features other than potential targets to ignore in regression. These are the keys in the data."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="mean_f1",
        help="Should we try to predict the mean F1 or the max F1 over references?"
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=10
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Regularization parameter for ridge regression"
    )
    parser.add_argument(
        "--baseline_constant",
        type=float,
        default=0.5,
        help="The constant value to predict as F1 for the constant baseline"
    )
    parser.add_argument(
        "--use_bucket_accuracy",
        action="store_true",
        help="Use bucket accuracy instead of MAE as the metric"
    )
    parser.add_argument(
        "--test_predictions_output",
        type=str,
        help="Output file for test predictions"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    test_predictions_file = open(args.test_predictions_output, "w") if args.test_predictions_output else None

    metric = bucket_accuracy if args.use_bucket_accuracy else mean_absolute_error

    data = [json.loads(line) for line in open(args.data)]
    features = []
    for key in data[0].keys():
        ignore_features = ["id", "pred_mean_f1", "pred_max_f1"]
        if args.ignore_features:
            ignore_features += args.ignore_features
        if key in ignore_features:
            continue
        features.append(key)

    print(f"Using features: {features}")

    if args.test_data:
        print("Test data is provided.")
        test_data = [json.loads(line) for line in open(args.test_data)]
        train_error, test_error, baseline_test_error = get_train_test_errors(data,
                                                                             test_data,
                                                                             metric,
                                                                             features,
                                                                             args.target,
                                                                             args.alpha,
                                                                             args.baseline_constant,
                                                                             test_predictions_file,
                                                                             args.verbose)
        print(f"Train error: {train_error}")
        print(f"Test error: {test_error}")
        for feature, error in baseline_test_error.items():
            print(f"{feature}: {error}")

    else:
        print("Test data is not provided. Will run cross validation on training data.")
        train_errors, test_errors, baseline_test_errors = run_cross_validation(data,
                                                                               args.folds,
                                                                               metric,
                                                                               features,
                                                                               args.target,
                                                                               args.alpha,
                                                                               args.baseline_constant,
                                                                               test_predictions_file,
                                                                               args.verbose)
        print(f"Average train error: {np.mean(train_errors)} (+/- {np.std(train_errors)})")
        print(f"Average test error: {np.mean(test_errors)} (+/- {np.std(test_errors)})")
        print("Average baseline test errors:")
        for key in baseline_test_errors:
            print(f"{key}: {np.mean(baseline_test_errors[key])}")

if __name__ == "__main__":
    main()
