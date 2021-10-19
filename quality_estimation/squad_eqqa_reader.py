import json
import logging
from typing import Optional
from collections import defaultdict
import random

from overrides import overrides

import torch

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import (
    MetadataField,
    TextField,
    TensorField
)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer


logger = logging.getLogger(__name__)

random.seed(20190)


@DatasetReader.register("squad_eqqa")
class SquadEqqaReader(DatasetReader):
    def __init__(
        self,
        transformer_model_name: str = "roberta-large",
        max_sequence_length: int = 512,
        padding_token: str = "<pad>",
        ratio_of_ones_to_keep: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._transformer_model_name = transformer_model_name
        self._tokenizer = PretrainedTransformerTokenizer(
            transformer_model_name, add_special_tokens=False
        )

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(transformer_model_name)
        }
        self._max_sequence_length = max_sequence_length
        self._padding_token = padding_token
        # This is the proportion of instances whose F1 is 1.0 that we'll keep. You want to set this to a lower
        # value if the dataset has too many instances with a perfect F1 score.
        self._ones_ratio = ratio_of_ones_to_keep
        self._stats = defaultdict(int)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading json file at %s", file_path)

        dataset = [json.loads(line) for line in open(file_path)]
        for datum in self.shard_iterable(dataset):
            if datum["max_f1"] == 1.0:
                yielding_probability = random.random()
                if yielding_probability > self._ones_ratio:
                    self._stats["number of instances with perfect F1 skipped"] += 1
                    continue

            self._stats["number of instances yielded"] += 1
            yield self.text_to_instance(datum["context"], datum["question"], datum["id"], datum["max_f1"])
 
        print("Dataset stats:\n")
        for key, value in self._stats.items():
            print(f"{key}: {value}")

    @overrides
    def text_to_instance(
            self,
            context: str,
            question: str,
            question_id: str,
            target_f1: int,
    ) -> Instance:
        tokenized_context = self._tokenizer.tokenize(context)
        tokenized_question = self._tokenizer.tokenize(question)
        allowed_context_length = (
                self._max_sequence_length
                - len(tokenized_question)
                - len(self._tokenizer.sequence_pair_mid_tokens)
                - len(self._tokenizer.sequence_pair_start_tokens)
                - len(self._tokenizer.sequence_pair_end_tokens)
        )

        if len(tokenized_context) > allowed_context_length:
            self._stats["truncated contexts"] += 1
            tokenized_context = tokenized_context[:allowed_context_length]

        tokenized_input = self._tokenizer.add_special_tokens(tokenized_question, tokenized_context)
        if len(tokenized_input) < self._max_sequence_length:
            original_question_context_length = len(tokenized_input)
            padding_length = self._max_sequence_length - original_question_context_length
            tokenized_input += [Token(self._padding_token)] * padding_length

        input_field = TextField(tokenized_input)
        target_field = TensorField(torch.tensor([target_f1]))
        metadata_field = MetadataField(
                {
                    "question": question,
                    "context": context,
                    "question_id": question_id,
                    "target_f1": target_f1
                }
        )
        return Instance(
                {
                    "question_and_context": input_field,
                    "target_f1": target_field,
                    "metadata": metadata_field
                }
        )

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["question_and_context"].token_indexers = self._token_indexers
