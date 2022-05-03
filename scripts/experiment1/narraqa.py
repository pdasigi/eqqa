from collections import Counter

import datasets
import numpy as np
print("datasets version:", datasets.__version__)


def read_mocha_split(parent_dir, split, dataset):
    import json
    _filepath = f"{parent_dir}/{split}.json"
    
    with open(_filepath) as f:
        data = json.load(f)
    return data[dataset]


if __name__ == "__main__":
    # --------------------------------------------------
    # Step 1. Load dataset from HuggingFace
    # --------------------------------------------------
    narrqa = datasets.load_dataset("narrativeqa")
    train, dev, test = [narrqa[split] for split in ("train", "validation", "test")]
    hf_ids = dev.map(lambda d: {"id": d["document"]["id"]}).unique("id")

    # --------------------------------------------------
    # Step 2. Load dataset from MOCHA
    # --------------------------------------------------
    mocha_dev = read_mocha_split("/home/kat/Projects/PhD/qasper-experiments/eqqa/data/mocha", "dev", "narrativeqa")
    mocha_ids = list(mocha_dev.keys())

    # ---------------------------------------------------
    # Step 3. Look for overlapping examples
    # ---------------------------------------------------
    hf_ids = Counter(hf_ids)
    mocha_ids = Counter(mocha_ids)
    print("ID overlap:", hf_ids & mocha_ids)

    hf_questions = dev.map(lambda d: {"question_text": d["question"]["text"]}).unique("question_text")
    mocha_questions = np.unique([ex["question"] for ex_id, ex in mocha_dev.items()]).tolist()

    hf_questions = Counter(hf_questions)
    mocha_questions = Counter(mocha_questions)
    print("Questions overlap:", len(hf_questions & mocha_questions), "out of", len(hf_questions))

    # Context
    hf_context = dev.map(lambda d: {"context": d["document"]["summary"]["text"]}).unique("context")
    mocha_context = np.unique([ex["context"] for ex_id, ex in mocha_dev.items()]).tolist()

    hf_context = Counter(hf_context)
    mocha_context = Counter(mocha_context)
    print("Context overlap:", len(hf_context & mocha_context), "out of", len(hf_context))