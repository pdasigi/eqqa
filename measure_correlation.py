import json
from tqdm import tqdm
from scipy import stats
import numpy as np
import sys
from allennlp_models.rc.metrics import SquadEmAndF1

metric = SquadEmAndF1()
num_samples = int(sys.argv[1])
print(f"Using {num_samples} samples for BALD")

def get_f1(pred, ans):
    metric(pred, [ans])
    _, f1 = metric.get_metric(True)
    return f1

predictions = json.load(open("squad_dev_outputs.json"))
predictions_ttd = json.load(open("squad_dev_ttd_outputs.json"))

max_f1s = []
mean_f1s = []
pairwise_ttd_f1s = []
ttd_pred_probability_mean = []
ttd_pred_probability_std = []

for prediction, prediction_ttd in tqdm(zip(predictions, predictions_ttd)):
    assert prediction["qid"] == prediction_ttd["qid"]
    pred = prediction["predictions"][0]
    answers = prediction["answers"]
    pred_f1s = [get_f1(pred, answer) for answer in answers]
    max_f1s.append(max(pred_f1s))
    mean_f1s.append(sum(pred_f1s)/len(pred_f1s))
    ttd_preds = prediction_ttd["predictions"][:num_samples]
    ttd_probs = prediction_ttd["probabilities"][:num_samples]
    ttd_pred_f1s = []
    for i in range(len(ttd_preds) - 1):
        for j in range(i + 1, len(ttd_preds)):
            ttd_pred_f1s.append(get_f1(ttd_preds[i], ttd_preds[j]))

    pairwise_ttd_f1s.append(sum(ttd_pred_f1s) / len(ttd_pred_f1s))
    ttd_pred_probability_mean.append(np.mean(ttd_probs))
    ttd_pred_probability_std.append(np.std(ttd_probs))

assert len(max_f1s) == len(mean_f1s) == len(pairwise_ttd_f1s)
max_f1_correl = stats.pearsonr(max_f1s, pairwise_ttd_f1s)
print("Correlation of TTD F1s with Max-F1 against references:", max_f1_correl)
mean_f1_correl = stats.pearsonr(mean_f1s, pairwise_ttd_f1s)
print("Correlation of TTD F1s with Mean-F1 against references:", mean_f1_correl)
mean_prob_correl = stats.pearsonr(mean_f1s, ttd_pred_probability_mean)
print("Correlation of TTD probability mean with Mean-F1 against references:", mean_prob_correl)
std_prob_correl = stats.pearsonr(mean_f1s, ttd_pred_probability_std)
print("Correlation of TTD probability mean with Mean-F1 against references:", std_prob_correl)
