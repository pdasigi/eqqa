from dict_utils import shuffle_dict

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


def create_dataset(path, split, output_dir, **kwargs):
    from metrics import compute_metrics

    data = read_json_dataset(path, split)
    compute_metrics(data, **kwargs)
    write_json_dataset(data, output_dir, f"{split}_metrics")
    

if __name__ == "__main__":
    # TODO - Replace w/ argparse
    ROOT_DIR = ".."
    ORIGINAL_MOCHA_DIR = f"{ROOT_DIR}/data/mocha"

    PREPROC_DIR = f"{ROOT_DIR}/data/lr_experiments"
    os.makedirs(PREPROC_DIR, exist_ok=True)

    W2VEC_MODEL = 'GoogleNews-vectors-negative300'
    W2VEC_PATH = f"{ROOT_DIR}/data/preprocessing/{W2VEC_MODEL}.bin.gz"

    default_kwargs = {
        "output_dir": PREPROC_DIR,
        "w2vec_path": W2VEC_PATH,
    }

    # Create raw metric modeling datasets (w/ different metrics)
    # create_dataset(ORIGINAL_MOCHA_DIR, "dev", **default_kwargs)
    # create_dataset(ORIGINAL_MOCHA_DIR, "test", **default_kwargs)
    # create_dataset(ORIGINAL_MOCHA_DIR, "train", **default_kwargs)

    ORIGINAL_MOCHA_DIR = f"{ROOT_DIR}/data/lr_experiments"
    PREPROC_DIR = f"{ROOT_DIR}/data/lr_experiments"
    os.makedirs(PREPROC_DIR, exist_ok=True)

    W2VEC_MODEL = 'GoogleNews-vectors-negative300'
    W2VEC_PATH = f"{ROOT_DIR}/data/preprocessing/{W2VEC_MODEL}.bin.gz"

    default_kwargs = {
        "output_dir": PREPROC_DIR,
        "w2vec_path": W2VEC_PATH,
    }

    # Create raw metric modeling datasets (w/ different metrics)
    create_dataset(ORIGINAL_MOCHA_DIR, "qasper", **default_kwargs)