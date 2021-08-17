import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--f1s", type=str)
    parser.add_argument(
        "--ignore_papers",
        type=str,
        help="Papers to ignore if f1 scores are from a model trained on a part of the training data"
    )
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    qasper_dataset = json.load(open(args.dataset))
    f1_values = json.load(open(args.f1s))
    f1_dict = {}
    ignore_papers = set()
    if args.ignore_papers:
        ignore_papers = set(json.load(open(args.ignore_papers)).keys())
    for datum in f1_values:
        f1_dict[datum["qid"]] = (datum["mean_f1"], datum["max_f1"])

    new_qasper_dataset = {}
    for paper_id, paper_info in qasper_dataset.items():
        if paper_id in ignore_papers:
            continue
        new_qas = []
        for qa_info in paper_info["qas"]:
            question_id = qa_info["question_id"]
            mean_f1, max_f1 = f1_dict[question_id]
            new_qas.append({"question": qa_info["question"],
                            "question_id": question_id,
                            "mean_f1": mean_f1,
                            "max_f1": max_f1})
        paper_info["qas"] = new_qas
        new_qasper_dataset[paper_id] = paper_info

    with open(args.output, "w") as outfile:
        json.dump(new_qasper_dataset, outfile, indent=2)


if __name__ == "__main__":
    main()
