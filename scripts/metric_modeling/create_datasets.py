import os
import json




def read_json_dataset(parent_dir: str, filename: str, dataset: str=None) -> dict:
    """Loads the dataset from the specified path. 
    
    It assumes the dataset is in JSON format and that is
    represented as {tag1: {examples}, tag2: {...}, ...}
    where tag1 and tag2 are dataset tags that the user
    can specify. If none are specified all the datasets
    will be returned.
    """
    data = json.load(open(f"{parent_dir}/{filename}.json"))
    
    if dataset is None:
        datasets = list(data.keys())
    else:
        datasets = dataset if isinstance(dataset, list) else [dataset]
    data = {d: datum for d, datum in data.items() if d in datasets}
    return data


def write_json_dataset(data, parent_dir, filename):
    output_file = f"{parent_dir}/{filename}.json"
    with open(output_file, 'w', encoding='utf-8') as writer:
        json.dump(data, writer, ensure_ascii=False, indent=2)
        

def create_dataset(mocha_dataset, parent_dir, dataset, split, **kwargs):
    from metrics import compute_metrics
    compute_metrics({dataset: mocha_dataset[dataset]}, **kwargs)
    
    new_examples = {}
    for example_id, example in mocha_dataset[dataset].items():
        example["dataset"] = dataset
        example["split"] = split
        new_examples[f"{dataset}-{example_id}"] = example
    
    write_json_dataset(new_examples, parent_dir, f"mocha_mmproc_{dataset}_{split}")


def create_dataset_from_MOCHA(path, split, **kwargs):
    data = read_json_dataset(path, split)
    datasets = list(data.keys())

    for dataset in datasets:
        print("=" * 30, dataset, "=" * 30)
        create_dataset(mocha_dataset=data, dataset=dataset, split=split, **kwargs)

        # Hopefully calls Garbage collection (since there is no longer a reference)
        d = data[dataset]
        data[dataset] = {}
        del d

    del data
    del datasets



if __name__ == "__main__":
    ROOT_DIR = "../.."
    ORIGINAL_MOCHA_DIR = f"{ROOT_DIR}/data/metric-modeling/mocha"

    PREPROC_DIR = f"{ROOT_DIR}/data/raw_splits"
    os.makedirs(PREPROC_DIR, exist_ok=True)

    W2VEC_MODEL = 'GoogleNews-vectors-negative300'
    W2VEC_PATH = f"../../data/preprocessing/{W2VEC_MODEL}.bin.gz"

    default_kwargs = {
        "parent_dir": PREPROC_DIR,
        "w2vec_path": W2VEC_PATH,
    }

    create_dataset_from_MOCHA(ORIGINAL_MOCHA_DIR, "train", **default_kwargs)