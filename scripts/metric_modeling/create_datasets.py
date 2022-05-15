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


def create_all_datasets_experiment(raw_dir, exp_dir, datasets, split, prefix="mocha_mmproc_"):
    final_dataset = {}
    
    for dataset in datasets:
        filename = f"{prefix}{dataset}_{split}"
        
        data = read_json_dataset(raw_dir, filename)
        final_dataset.update(data)
    
    final_dataset = shuffle_dict(final_dataset)
    write_json_dataset(final_dataset, exp_dir, split)


def create_loov_experiment(raw_dir, exp_dir, datasets, split, prefix="mocha_mmproc_"):
    def _get_dataset(datasets):
        final_dataset = {}

        for dataset in datasets:
            filename = f"{prefix}{dataset}_{split}"
            data = read_json_dataset(raw_dir, filename)
            final_dataset.update(data)

        return shuffle_dict(final_dataset)
    
    for i, _ in enumerate(datasets):
        loo_dataset = datasets[i]
        final_dataset = _get_dataset(datasets[:i] + datasets[i+1:])
        write_json_dataset(final_dataset, exp_dir, f"except_{loo_dataset}_{split}")
    
        data = read_json_dataset(raw_dir, f"{prefix}{loo_dataset}_{split}")
        write_json_dataset(data, exp_dir, f"{loo_dataset}_{split}")



if __name__ == "__main__":
    # TODO - Replace w/ argparse
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

    # Create raw metric modeling datasets (w/ different metrics)
    create_dataset_from_MOCHA(ORIGINAL_MOCHA_DIR, "dev", **default_kwargs)
    create_dataset_from_MOCHA(ORIGINAL_MOCHA_DIR, "test", **default_kwargs)
    create_dataset_from_MOCHA(ORIGINAL_MOCHA_DIR, "train", **default_kwargs)

    # We are interested in two types of experiments
    # All datasets (AD): train over all datasets and evaluate in all others
    # Leave one out (LOOV): train in all datasets except one, evaluate in that one.

    # Prepare datasets for AD experiments
    DATASETS = ("cosmosqa", "drop", "mcscript", "narrativeqa", "quoref", "socialiqa")
    create_all_datasets_experiment(PREPROC_DIR, f"{PREPROC_DIR}/all_datasets", DATASETS, "dev")
    create_all_datasets_experiment(PREPROC_DIR, f"{PREPROC_DIR}/all_datasets", DATASETS, "test")
    create_all_datasets_experiment(PREPROC_DIR, f"{PREPROC_DIR}/all_datasets", DATASETS, "train")

    # Prepare datastets for leave-one-out experiment
    create_loov_experiment(PREPROC_DIR, f"{PREPROC_DIR}/loov_datasets", DATASETS, "dev")
    create_loov_experiment(PREPROC_DIR, f"{PREPROC_DIR}/loov_datasets", DATASETS, "test")
    create_loov_experiment(PREPROC_DIR, f"{PREPROC_DIR}/loov_datasets", DATASETS, "train")
