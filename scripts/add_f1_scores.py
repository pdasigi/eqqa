import json
import sys
import numpy as np
from allennlp_models.rc.metrics import SquadEmAndF1

mean_f1s = []
max_f1s = []

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

data = json.load(open(sys.argv[1]))
for datum in data:
    prediction = datum["predictions"][0]
    references = datum["answers"]
    f1s = [get_f1(prediction, reference) for reference in references]
    mean_f1 = np.mean(f1s)
    max_f1 = max(f1s)
    mean_f1s.append(mean_f1)
    max_f1s.append(max_f1)
    datum["mean_f1"] = mean_f1
    datum["max_f1"] = max_f1

with open(sys.argv[1], "w") as outfile:
    json.dump(data, outfile, indent=2)


print(f"Average mean f1: {np.mean(mean_f1s)}")
print(f"Average max f1: {np.mean(max_f1s)}")
