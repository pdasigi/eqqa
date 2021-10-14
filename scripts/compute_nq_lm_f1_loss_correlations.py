import argparse
import json
import math
from transformers import AutoTokenizer
from scipy.stats import spearmanr as correl

TOKENIZER = AutoTokenizer.from_pretrained('gpt2')


def get_num_tokens(question, context):
    question_tokens = TOKENIZER.tokenize(question)
    context_tokens = TOKENIZER.tokenize(context)
    # +1 for separator
    return min(len(question_tokens) + len(context_tokens) + 1, 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression_output", type=str, required=True)
    parser.add_argument("--lm_losses", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    regression_output_data = [json.loads(line) for line in open(args.regression_output)]
    regression_losses = {datum["id"]: abs(datum["prediction"] - datum["target"])
                         for datum in regression_output_data}
    lm_losses_data = [json.loads(line) for line in open(args.lm_losses)]
    lm_losses = {datum["question_ids"][0]: datum["loss"] for datum in lm_losses_data}
    normalized_lm_losses = {}
    dataset = json.load(open(args.data))
    for datum in dataset:
        num_tokens = get_num_tokens(datum["question"], datum["context"])
        normalized_lm_losses[datum["id"]] = lm_losses[datum["id"]] / num_tokens

    ids_list = list(regression_losses.keys())
    regression_lm_correl = correl([regression_losses[id_] for id_ in ids_list],
                                  [math.exp(-lm_losses[id_]) for id_ in ids_list])
    print(f"Correlation between regression losses and LM losses: {regression_lm_correl}")
    regression_norm_lm_correl = correl([regression_losses[id_] for id_ in ids_list],
                                       [math.exp(-normalized_lm_losses[id_]) for id_ in ids_list])
    print(f"Correlation between regression losses and normalized LM losses: {regression_norm_lm_correl}")


if __name__ == "__main__":
    main()

