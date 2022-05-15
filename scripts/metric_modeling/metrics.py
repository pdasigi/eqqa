from pycocoevalcap.meteor.meteor import Meteor as pccMeteor
from pycocoevalcap.rouge.rouge import Rouge as pccRouge
from pycocoevalcap.bleu.bleu import Bleu as pccBleu
from bert_score import score as BERT_SCORE
from datasets import load_metric

from dict_utils import update_examples
from tokenization import normalize_answer, remove_punc
from typing import Any, Dict, List

Example = Dict[str, Any]


def add_bleu(mocha_dataset: Dict[str, List[Example]], order: int=4):
    BLEU = pccBleu(order)

    for dataset, examples in mocha_dataset.items():
        refs = {i: [remove_punc(instance['reference'])] for i, instance in
                enumerate(examples.values())}
        cands = {i: [remove_punc(instance['candidate'])] for i, instance in
                 enumerate(examples.values())}
        
        # compute_scores return (aggregate-bleu, instance-wise bleu)
        # -- by accessing the first index, we get the bleu per instance
        bleu_scores = BLEU.compute_score(refs, cands, verbose=0)[1]
        
        for i in range(order):
            update_examples(examples.values(), f"bleu{i+1}", bleu_scores[i])
            
def add_bleu_hf(mocha_dataset, order: int=4, prefix="hf_"):
    BLEU = load_metric("bleu", keep_in_memory=True)
    for dataset, examples in mocha_dataset.items():
        invalid_examples = set()

        for example_id, example in examples.items():
            candidate = normalize_answer(example["candidate"]).split()
            reference = normalize_answer(example["reference"]).split()
            
            for i in range(1, order+1):
                try:
                    scores = BLEU.compute(predictions=[candidate],
                                          references=[[reference]],
                                          max_order=i)["bleu"]
                
                except:
                    invalid_examples.add(example_id)
                    scores = -1

                example[f"{prefix}bleu{i}"] = scores
                
        print("Dropping", len(invalid_examples), "examples:", invalid_examples, "from dataset", dataset)

        for invalid_id in invalid_examples:
            print(invalid_id, examples.pop(invalid_id))


def add_meteor(mocha_dataset):
    METEOR = pccMeteor()

    for dataset, examples in mocha_dataset.items():
        refs = {i: [remove_punc(instance['reference'])] for i, instance in
                enumerate(examples.values())}
        cands = {i: [remove_punc(instance['candidate'])] for i, instance in
                 enumerate(examples.values())}
        pred_scores = METEOR.compute_score(refs, cands)[1]
        update_examples(examples.values(), "meteor", pred_scores)


def add_rouge(mocha_dataset):
    ROUGE = pccRouge()

    for dataset, examples in mocha_dataset.items():
        refs = {i: [remove_punc(instance['reference'])] for i, instance in
                enumerate(examples.values())}
        cands = {i: [remove_punc(instance['candidate'])] for i, instance in
                 enumerate(examples.values())}
        pred_scores = ROUGE.compute_score(refs, cands)[1]
        update_examples(examples.values(), "rougeL", pred_scores)

        
def add_bertscore(mocha_dataset):
    for dataset, examples in mocha_dataset.items():
        refs = [remove_punc(instance['reference']) for instance in examples.values()]
        cands = [remove_punc(instance['candidate']) for instance in examples.values()]
        pred_scores = BERT_SCORE(cands, refs, lang='en')[-1].tolist()
        update_examples(examples.values(), "bertscore", pred_scores)


def add_bleurt(mocha_dataset):
    BLEURT = load_metric("bleurt", keep_in_memory=True)

    for dataset, examples in mocha_dataset.items():
        for example in examples.values():
            candidate = normalize_answer(example["candidate"])
            reference = normalize_answer(example["candidate"])
            scores = BLEURT.compute(predictions=[candidate],
                                    references=[reference])
            example["bleurt"] = scores["scores"][0]


def add_char_edit_rate(mocha_dataset):
    """Compute word edit rate. 
    
    The formula is like the character_edit_rate but using words
    rather than characters.
    """
    # https://github.com/huggingface/datasets/tree/fad939b5e17b672a4eda7de2cd8e24d98f3d5b26/metrics/wer
    # !pip install jiwer
    CER = load_metric("cer", keep_in_memory=True)
    
    for dataset, examples in mocha_dataset.items():
        for example in examples.values():
            candidate = normalize_answer(example["candidate"])
            reference = normalize_answer(example["reference"])

            scores = CER.compute(predictions=[candidate], references=[reference])
            example["char_edit_score"] = scores


def add_word_edit_rate(mocha_dataset):
    """Compute word edit rate. 
    
    The formula is like the character_edit_rate but using words
    rather than characters.
    """
    # https://github.com/huggingface/datasets/tree/fad939b5e17b672a4eda7de2cd8e24d98f3d5b26/metrics/wer
    # !pip install jiwer
    WER = load_metric("wer", keep_in_memory=True)
    
    for dataset, examples in mocha_dataset.items():
        for example in examples.values():
            candidate = normalize_answer(example["candidate"])
            reference = normalize_answer(example["reference"])

            scores = WER.compute(predictions=[candidate], references=[reference])
            example["word_edit_score"] = scores
    
def add_recall(mocha_dataset):
    from collections import Counter

    for dataset, examples in mocha_dataset.items():
        for example in examples.values():
            candidate = normalize_answer(example["candidate"]).split()
            reference = normalize_answer(example["reference"]).split()

            true_tks, pred_tks = Counter(reference), Counter(candidate)
        
            tp = sum((true_tks & pred_tks).values())
            
            if tp == 0:
                example["recall"] = 0
            else:
                example["recall"] = tp / len(reference)

            example["tp"] = tp
            example["fn"] = len(reference) - tp


def add_precision(mocha_dataset):
    from collections import Counter

    for dataset, examples in mocha_dataset.items():
        for example in examples.values():
            candidate = normalize_answer(example["candidate"]).split()
            reference = normalize_answer(example["reference"]).split()

            true_tks, pred_tks = Counter(reference), Counter(candidate)
        
            tp = sum((true_tks & pred_tks).values())
            example["precision"] = 0 if tp == 0 else tp / len(candidate)

            example["tp"] = tp
            example["fp"] = len(candidate) - tp


def add_f_score(mocha_dataset, beta=1):
    f_name = f"f{beta}_score"
    for dataset, examples in mocha_dataset.items():
        for example in examples.values():
            recall = example["recall"]
            precis = example["precision"]
            
            if precis == 0 or recall == 0:
                example[f_name] = 0
            else:
                beta = beta*beta
                num = precis * recall
                den = (beta * precis + recall)
                example[f_name] = (1+beta) * num / den


def add_rouge_order_n(mocha_dataset, rouge_types, use_stemmer=False, prefix="hf_"):
    ROUGE = load_metric("rouge", keep_in_memory=True) 
    #^Note: requires installing rouge-score (!pip install rouge-score)
    # https://github.com/huggingface/datasets/issues/617

    for dataset, examples in mocha_dataset.items():
        candidate = [[normalize_answer(ex["candidate"])] for ex in examples.values()]
        reference = [[normalize_answer(ex["reference"])] for ex in examples.values()]

        scores = ROUGE.compute(predictions=candidate,
                               references=reference,
                               use_stemmer=use_stemmer,
                               use_aggregator=False)

        for rouge_type in rouge_types:
            rouge_scores = [s.fmeasure for s in scores[rouge_type]]
            update_examples(examples.values(), prefix + rouge_type, rouge_scores)


def add_sari(mocha_dataset, source_col="context"):
    """Compute the SARI score.
    
    System output against references and against the input sentence.
    Often used for evaluating automatic text simplification systems.    
    https://github.com/huggingface/datasets/tree/master/metrics/sari.
    
    The range of values for the SARI score is between 0 and 100 -- the
    higher the value, the better the performance of the model being
    evaluated, with a SARI of 100 being a perfect score.
    
    We divide the score by 100 to be on the range (0, 1).
    This score is computed as: 
        sari = ( F1_add + F1_keep + P_del) / 3,
    where
    - F1_add is the n-gram F1 score for add operations
    - F1_keep is the n-gram F1 score for keep operations
    - P_del is the n-gram precision score for delete operations

    The number of n grams, n, is equal to 4, as in the original paper.
    """
    SARI = load_metric("sari", keep_in_memory=True)
    
    for dataset, examples in mocha_dataset.items():
        for example in examples.values():
            sources = normalize_answer(example[source_col])
            candidate = normalize_answer(example["candidate"])
            reference = normalize_answer(example["reference"])

            scores = SARI.compute(sources=[sources], predictions=[candidate], references=[[reference]])
            example[f"sari_{source_col}"] = scores["sari"] / 100


def add_prec_rec_at_error1(mocha_dataset):
    """Computes the fraction of correct tokens in prediction until first error."""
    def _get_metric_at_1(reference, candidate):
        if candidate != reference:
            for i, tokens in enumerate(zip(candidate, reference)):
                cand_tk, ref_tk = tokens

                if cand_tk != ref_tk:
                    return 0 if i == 0 or len(reference) == 1 else i/(len(reference)-1)
        return 1
            
    
    for dataset, examples in mocha_dataset.items():
        for example in examples.values():
            candidate = normalize_answer(example["candidate"]).split()
            reference = normalize_answer(example["reference"]).split()
            
            example["precision_at_err1"] = _get_metric_at_1(candidate, reference)
            example["recall_at_err1"] = _get_metric_at_1(reference, candidate)


def add_word_movers_distance(mocha_dataset, w2vec_path):
    # https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html
    from gensim.models import KeyedVectors
    model = KeyedVectors.load_word2vec_format(w2vec_path, binary=True)
        
    for dataset, examples in mocha_dataset.items():
        for example in examples.values():
            candidate = normalize_answer(example["candidate"], remove_stopwords=True)
            reference = normalize_answer(example["reference"], remove_stopwords=True)
            score = model.wmdistance(candidate, reference)
            
            example["wmd"] = score


def compute_metrics(data, w2vec_path: str):
    print(">", "BLEU")
    add_bleu(data)
    add_bleu_hf(data)

    print(">", "ROUGE")
    add_rouge(data)
    add_rouge_order_n(data, ["rouge1", "rouge2", "rougeL", "rougeLsum"])

    print(">", "METEOR")
    add_meteor(data)

    print(">", "TOKEN-OVERLAP")
    add_recall(data)
    add_precision(data)
    add_f_score(data)
    add_prec_rec_at_error1(data)
    
    print(">", "EDIT RATES")
    add_char_edit_rate(data)
    add_word_edit_rate(data)
    add_sari(data, "context")
    add_sari(data, "question")

    print(">", "Learned Metrics")
    add_bertscore(data)
    add_bleurt(data)

    # https://radimrehurek.com/gensim/models/word2vec.html#pretrained-models
    # >>> pip install gdown
    # >>> gdown 0B7XkCwpI5KDYNlNUTTlSS21pQmM
    add_word_movers_distance(data, w2vec_path)