import json
import argparse
from tqdm import tqdm
from allennlp_models.rc.predictors import TransformerQAPredictor
from allennlp_models.rc.dataset_readers import TransformerSquadReader
from allennlp.models.archival import load_archive
from datasets import load_dataset


def activate_dropouts(model):
    model.eval()
    num_dropout_masks = 0
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            num_dropout_masks += 1
            module.train()
    print(f"Activated {num_dropout_masks} masks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--model",
            type=str,
            help="Location of serialized model",
            default="/net/nfs2.allennlp/pradeepd/data/squad_models/transformer-qa.2021-02-12.tar.gz"
    )
    parser.add_argument(
            "--passes",
            type=int,
            help="Number of test-time dropout passes",
            default=20
    )
    parser.add_argument(
            "--output",
            type=str,
            help="Location of output file",
            default="data/squad_dev_ttd_outputs.json"
    )
    parser.add_argument(
            "--no-bald",
            action="store_true",
            help="Will not activate test-time dropout and will run only one pass if this is set",
            dest="no_bald"
    )
    args = parser.parse_args()
    archive = load_archive(args.model)
    archive.model.cuda()

    predictor = TransformerQAPredictor(archive.model, archive.dataset_reader)
    if not args.no_bald:
        activate_dropouts(predictor._model)

    num_passes = 1 if args.no_bald else args.passes 
    dataset = load_dataset('squad_v2', split='validation')
    output_data = []

    for datum in tqdm(dataset):
        outputs = [predictor.predict(question=datum['question'], passage=datum['context']) for _ in range(num_passes)]
        predictions = [o['best_span_str'] for o in outputs]
        prediction_probabilities = [o['best_span_probs'] for o in outputs]
        output_data.append(
                {
                    'qid': datum['id'],
                    'answers': datum['answers']['text'],
                    'predictions': predictions,
                    'probabilities': prediction_probabilities,
                }
        )

    with open(args.output, 'w') as outfile:
        json.dump(output_data, outfile, indent=2)

if __name__ == "__main__":
    main()
