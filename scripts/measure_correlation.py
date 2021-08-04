import json
from tqdm import tqdm
from scipy import stats
import numpy as np
import argparse
from allennlp_models.rc.metrics import SquadEmAndF1


def main():
    metric = SquadEmAndF1()
    f1_hash = {}
    def get_f1(pred, ans):
        pred = pred.strip()
        ans = ans.strip()
        if (pred, ans) in f1_hash:
            return f1_hash[(pred, ans)]
        if (ans, pred) in f1_hash:
            return f1_hash[(ans, pred)]
        metric(pred, [ans])
        _, f1 = metric.get_metric(True)
        f1_hash[(pred, ans)] = f1
        return f1

    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, help="Number of MC dropout samples to use for BALD", default=10)
    parser.add_argument("--predictions", type=str, help="JSON file with original predictions")
    parser.add_argument("--ttd_predictions", type=str,
                        help="JSON file with predictions from test time dropout ensemble")
    parser.add_argument("--ignore-no-answers", action="store_true", dest="ignore_no_answers")
    parser.add_argument("--output", type=str, help="Location of an output file to write features")
    args = parser.parse_args()
    output_file = open(args.output, "w") if args.output else None
    predictions = json.load(open(args.predictions))
    predictions_ttd = json.load(open(args.ttd_predictions))
    num_samples = args.samples
    print(f"Using {num_samples} samples for BALD")

    max_f1s = []
    mean_f1s = []
    pairwise_ttd_f1s = []
    pairwise_ttd_f1_std = []
    against_max_ttd_f1s = []

    have_probabilities = "probabilities" in predictions[0]

    if have_probabilities:
        pred_probability = []
        ttd_pred_probability_mean = []
        ttd_pred_probability_std = []

    for prediction, prediction_ttd in tqdm(zip(predictions, predictions_ttd)):
        assert prediction["qid"] == prediction_ttd["qid"]
        pred = prediction["predictions"][0]
        answers = prediction["answers"]
        if not answers:
            if args.ignore_no_answers:
                continue
            answers = [""]

        pred_f1s = [get_f1(pred, answer) for answer in answers]
        max_f1 = max(pred_f1s)
        max_f1s.append(max_f1)
        mean_f1 = np.mean(pred_f1s)
        mean_f1s.append(mean_f1)
        ttd_preds = prediction_ttd["predictions"][:num_samples]
        ttd_pred_f1s = []
        against_max_ttd_pred_f1s = []
        for i in range(len(ttd_preds)):
            i_ttd_pred_f1s = []
            for j in range(len(ttd_preds)):
                if i == j:
                    continue
                i_j_f1 = get_f1(ttd_preds[i], ttd_preds[j])
                ttd_pred_f1s.append(i_j_f1)
                i_ttd_pred_f1s.append(i_j_f1)
            against_max_ttd_pred_f1s.append(max(i_ttd_pred_f1s))

        ttd_f1_mean = np.mean(ttd_pred_f1s)
        ttd_f1_std = np.std(ttd_pred_f1s)
        pairwise_ttd_f1s.append(ttd_f1_mean)
        pairwise_ttd_f1_std.append(ttd_f1_std)
        against_max_ttd_f1s.append(np.mean(against_max_ttd_pred_f1s))

        if have_probabilities:
            pred_probability.append(prediction["probabilities"][0])
            ttd_probs = prediction_ttd["probabilities"][:num_samples]
            ttd_pred_prob_mean = np.mean(ttd_probs)
            ttd_pred_prob_std = np.std(ttd_probs)
            ttd_pred_probability_mean.append(ttd_pred_prob_mean)
            ttd_pred_probability_std.append(ttd_pred_prob_std)
        if output_file is not None:
            data_to_dump = {"id": prediction["qid"],
                            "pred_mean_f1": mean_f1,
                            "pred_max_f1": max_f1,
                            "ttd_pairwise_f1_mean": ttd_f1_mean,
                            "ttd_pairwise_f1_std": ttd_f1_std}
            if have_probabilities:
                data_to_dump.update({"pred_probability": prediction["probabilities"][0],
                                     "ttd_prob_mean": ttd_pred_prob_mean,
                                     "ttd_prob_std": ttd_pred_prob_std})
            print(json.dumps(data_to_dump), file=output_file)

    assert len(max_f1s) == len(mean_f1s) == len(pairwise_ttd_f1s)
    max_f1_pairwise_correl = stats.pearsonr(max_f1s, pairwise_ttd_f1s)
    print("Correlation of Pairwise TTD F1s with Max-F1 against references:", max_f1_pairwise_correl)
    max_f1_max_correl = stats.pearsonr(max_f1s, against_max_ttd_f1s)
    print("Correlation of Against Max TTD F1s with Max-F1 against references:", max_f1_max_correl)
    mean_f1_pairwise_correl = stats.pearsonr(mean_f1s, pairwise_ttd_f1s)
    print("Correlation of Pairwise TTD F1s with Mean-F1 against references:", mean_f1_pairwise_correl)
    mean_f1_pairwise_std_correl = stats.pearsonr(mean_f1s, pairwise_ttd_f1_std)
    print("Correlation of STD of Pairwise TTD F1s with Mean-F1 against references:", mean_f1_pairwise_std_correl)

    if have_probabilities:
        mean_prob_correl = stats.pearsonr(mean_f1s, ttd_pred_probability_mean)
        print("Correlation of TTD probability mean with Mean-F1 against references:", mean_prob_correl)
        mean_prob_max_f1_correl = stats.pearsonr(max_f1s, ttd_pred_probability_mean)
        print("Correlation of TTD probability mean with Max-F1 against references:", mean_prob_max_f1_correl)
        std_prob_correl = stats.pearsonr(mean_f1s, ttd_pred_probability_std)
        print("Correlation of TTD probability STD with Mean-F1 against references:", std_prob_correl)
        prob_max_f1_correl = stats.pearsonr(max_f1s, pred_probability)
        print("Correlation of prediction probability with Max-F1 against references:", prob_max_f1_correl)
        prob_mean_f1_correl = stats.pearsonr(mean_f1s, pred_probability)
        print("Correlation of prediction probability with Mean-F1 against references:", prob_mean_f1_correl)


if __name__ == "__main__":
    main()
