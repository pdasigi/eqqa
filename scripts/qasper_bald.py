import json
import argparse
from tqdm import tqdm
from qasper_baselines.predictor import QasperPredictor
from qasper_baselines.dataset_reader import QasperReader, AnswerType
from qasper_baselines.model import QasperBaseline
from allennlp.models.archival import load_archive
from datasets import load_dataset


def extract_answers(reference_datum):
    answers = []
    for reference in reference_datum:
        if reference["unanswerable"]:
            answers.append({"text": "Unanswerable", "type": AnswerType.NONE})
        elif reference["extractive_spans"]:
            answers.append({"text": ", ".join(reference["extractive_spans"]),
                            "type": AnswerType.EXTRACTIVE})
        elif reference["free_form_answer"] != "":
            answers.append({"text": reference["free_form_answer"],
                            "type": AnswerType.ABSTRACTIVE})
        elif reference["yes_no"]:
            answers.append({"text": "Yes", "type": AnswerType.BOOLEAN})
        else:
            answers.append({"text": "No", "type": AnswerType.BOOLEAN})
    return answers


def get_paragraphs_from_full_text(full_text):
    paragraphs_to_return = []
    for section_name, paragraphs in zip(full_text['section_name'], full_text['paragraphs']):
        if section_name is not None:
            paragraphs_to_return.append(section_name)
        for paragraph in paragraphs:
            paragraph_text = paragraph.replace("\n", " ").strip()
            if paragraph_text:
                paragraphs_to_return.append(paragraph_text)
    return paragraphs_to_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--model",
            type=str,
            help="Location of serialized model",
            required=True
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
            default="data/qasper_dev_ttd_outputs.json"
    )
    parser.add_argument(
            "--no-bald",
            action="store_true",
            help="Will not activate test-time dropout and will run only one pass if this is set",
            dest="no_bald"
    )
    args = parser.parse_args()
    archive = load_archive(args.model)
    #archive.model.cuda()

    predictor = QasperPredictor(archive.model, archive.dataset_reader)
    if not args.no_bald:
        print("Setting model.transformer.train()")
        predictor._model.transformer.train()

    num_passes = 1 if args.no_bald else args.passes 
    dataset = load_dataset('qasper', split='validation')
    output_data = []

    for datum in tqdm(dataset):
        paragraphs = get_paragraphs_from_full_text(datum['full_text'])
        for question_id, question, references in zip(datum['qas']['question_id'],
                                                     datum['qas']['question'],
                                                     datum['qas']['answers']):
            answers = extract_answers(references['answer'])
            instance = archive.dataset_reader.text_to_instance(question,
                                                               paragraphs,
                                                               answer=answers[0]['text'],
                                                               additional_metadata={
                                                                   "all_answers": answers})
            outputs = [predictor.predict_instance(instance) for _ in range(num_passes)]
            predictions = [o['predicted_answers'] for o in outputs]
            output_data.append(
                    {
                        'qid': question_id,
                        'answers': [a["text"] for a in answers],
                        'predictions': predictions,
                    }
            )

    with open(args.output, 'w') as outfile:
        json.dump(output_data, outfile, indent=2)

if __name__ == "__main__":
    main()
