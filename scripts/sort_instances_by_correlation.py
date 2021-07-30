import json
from tqdm import tqdm
import argparse
from allennlp_models.rc.metrics import SquadEmAndF1

def main():
    metric = SquadEmAndF1()
    
    f1_hash = {}
    def get_f1(pred, ans):
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
    parser.add_argument("--output", type=str, help="Output file")
    args = parser.parse_args()
    predictions = json.load(open(args.predictions))
    predictions_ttd = json.load(open(args.ttd_predictions))
    num_samples = args.samples
    print(f"Using {num_samples} samples for BALD")

    output_data = []
    for prediction, prediction_ttd in tqdm(zip(predictions, predictions_ttd)):
        assert prediction["qid"] == prediction_ttd["qid"]
        pred = prediction["predictions"][0]
        answers = prediction["answers"] if prediction["answers"] else [""]
        pred_f1s = [get_f1(pred, answer) for answer in answers]
        mean_pred_f1 = sum(pred_f1s) / len(pred_f1s)
        ttd_preds = prediction_ttd["predictions"][:num_samples]
        ttd_pred_f1s = []
        for i in range(len(ttd_preds)):
            for j in range(len(ttd_preds)):
                if i == j:
                    continue
                i_j_f1 = get_f1(ttd_preds[i], ttd_preds[j])
                ttd_pred_f1s.append(i_j_f1)

        mean_ttd_pred_f1 = sum(ttd_pred_f1s) / len(ttd_pred_f1s)
        squared_error = (mean_pred_f1 - mean_ttd_pred_f1) ** 2
        output_data.append((squared_error, {"qid": prediction["qid"],
                                            "predictions": prediction["predictions"],
                                            "answers": prediction["answers"],
                                            "prediction_f1": mean_pred_f1,
                                            "sample_f1": mean_ttd_pred_f1}))

    data_to_dump = [x[1] for x in sorted(output_data, key=lambda y: y[0], reverse=True)]
    with open(args.output, "w") as outfile:
        json.dump(data_to_dump, outfile, indent=2)

if __name__ == "__main__":
    main()
