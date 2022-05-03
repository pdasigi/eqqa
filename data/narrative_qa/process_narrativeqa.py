"""Script takes the narrativeqa dataset and generates its splits.

Note this script is highly inspired by UnifiedQA's narrative_qa preprocessing
but encloses the following differences:

1. Creates json file (just like MOCHA) in the specified output_dir.
2. Does not add model-specific formatting (e.g., '\\n') but rather exposes
the different attributes: context, question, reference.


References
- https://github.com/allenai/unifiedqa/blob/master/encode_datasets.py#L822-L863
"""
from collections import defaultdict 
from tqdm import tqdm

import csv
import json
import numpy as np 


def generate_uuid(content, indent: int = 2) -> str:
    """Deterministic uuid generator of the `content`."""
    import hashlib
    content = json.dumps(content, sort_keys=True, indent=indent).encode("utf-8")
    return hashlib.md5(content).hexdigest()


# Write files
def write_json(out, content, mode="w+"):
    print("Writing to", out)
    with open(out, mode, encoding="utf-8") as fout:
        fout.write(json.dumps(content, indent=2, ensure_ascii=False))


def narrative_qa(input_dir, output_dir):
    paragraphs = {}
    # -----------------------------------------------------------
    # Summaries is a csv organized as follows:
    # -----------------------------------------------------------
    # - document_id: the id of the document
    # - split: the corresponding split (train, test, valid)
    # - summary: the text summary
    # - tokenized summary: the tokenized version of the summary
    # -----------------------------------------------------------
    with open(f"{input_dir}/summaries.csv") as f:
        spamreader = csv.reader(f)
        for i, line in enumerate(spamreader):
            if i == 0: # Skip header
                print(line)
                continue
            doc_id, summary = line[0], line[2]
            paragraphs[doc_id] = summary.replace("\n", "") # Inline paragraphs

    # -----------------------------------------------------------
    # qaps is a csv organized as follows:
    # -----------------------------------------------------------
    # - document_id: the id of the document
    # - split: the corresponding split (train, test, valid)
    # - question: the question
    # - answer1: answer 1
    # - answer2: answer2
    # and their tokenized versions..
    # -----------------------------------------------------------
    counts = {"train": 0, "dev": 0, "test": 0}
    train_dataset, dev_dataset, test_dataset = {}, {}, {}
    with open(f"{input_dir}/qaps.csv") as f:
        spamreader = csv.reader(f)
        for i, line in enumerate(spamreader):
            doc_id, split, question = line[0], line[1], line[2]
 
            if i == 0:
                print(line)
                continue

            examples = [{
                "doc_id": doc_id,
                "context": paragraphs[doc_id],
                "question": question,
                "reference": line[answer],
            } for answer in range(3, 5)]
            examples = {generate_uuid(ex): ex for ex in examples}

            if split == "train":
                dataset = train_dataset
            elif split == "test":
                dataset = test_dataset
            elif split == "valid":
                dataset = dev_dataset
                split = "dev"
            else:
                print(" >>>> ERROR ")
                continue

            dataset.update(examples)
            counts[split] += len(examples)

    write_json(f"{output_dir}/test.json", test_dataset)
    write_json(f"{output_dir}/train.json", train_dataset)
    write_json(f"{output_dir}/dev.json", dev_dataset)
    write_json(f"{output_dir}/counts.json", counts)


def generate_candidates_uqa(input_dir, output_dir, model_name, filename, n=None, **kwargs):
    def apply_unifiedqa_format(example):
        def normalize_text(text, col):
            """Lowercase and remove quotes."""
            import re
            text[col] = text[col].lower()
            text[col] = re.sub("'(.*)'", r"\1", text[col])

        def add_question_mark(example, col):
            example[col] = example[col].rstrip()
            if not example[col].endswith("?"):
                example[col] += "?"
        
        example = example.copy()
        normalize_text(example, "reference")
        normalize_text(example, "context")
        normalize_text(example, "question")
        add_question_mark(example, "question")
        
        return {
            "input_id": f"{example['question']} \\n {example['context']}", 
            "target": example["reference"]
        }

    def run_model(input_string, **generator_args):
        input_ids = tokenizer.encode(input_string, return_tensors="pt")
        res = model.generate(input_ids, **generator_args)
        return tokenizer.batch_decode(res, skip_special_tokens=True)

    dataset = f"{input_dir}/train.json"
    print("Reading the dataset from", dataset)
    with open(dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Getting dataset ready for model", model_name)
    for example in data.values():
        example.update(apply_unifiedqa_format(example))

    from transformers import T5Tokenizer, T5ForConditionalGeneration
    print("Loading model", model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    output_filepath = f"{output_dir}/train_{filename}.json"
    print("Starting generation at", output_filepath)
    final_dataset = defaultdict(dict)
    for i, (example_id, example) in tqdm(enumerate(data.items())):
        candidate_answers = run_model(example["input_id"], **kwargs)
        candidate_answers = np.unique(candidate_answers).tolist()
        candidate_answers = list(filter(lambda s: s != example["target"], candidate_answers))

        print("Example",i, "generated", len(candidate_answers))
        for candidate_answer in candidate_answers:
            new_example = {
                "example_id": example_id,
                "context": example["context"],
                "question": example["question"],
                "reference": example["reference"],
                "candidate": candidate_answer,
                "model": model_name,
            }
            new_example_id = generate_uuid(new_example)
            final_dataset[new_example_id] = new_example

        # FIXME
        # - Some instances may not be actually added to the dataset
        # - Add diversity penalty (?)
        # - Add perturbation to generated answers or confidence threshold (?)
        if i % 100 == 0:
            write_json(output_filepath, final_dataset, mode="a+")
            final_dataset = defaultdict(dict) # works like a buffer

        if n is not None and i > n:
            if len(final_dataset) != 0:
                write_json("out", final_dataset, mode="a+")
            break

    print("Generation finished...")


if __name__ == "__main__":
    input_dir = "/home/kat/Projects/PhD/qasper-experiments/eqqa/data/narrative_qa/raw"
    output_dir = "/home/kat/Projects/PhD/qasper-experiments/eqqa/data/narrative_qa/preprocessed"
    # narrative_qa(input_dir=input_dir, output_dir=output_dir)

    # Generation
    input_dir = output_dir
    generation_kwargs = {
        "num_beams": 25,
        "num_return_sequences": 20,
        "do_sample": False,
    }
    generate_candidates_uqa(
        input_dir,
        output_dir,
        model_name="allenai/unifiedqa-t5-small", 
        filename="unifiedqa-t5-small-20",
        n=20,
        **generation_kwargs,
    )
