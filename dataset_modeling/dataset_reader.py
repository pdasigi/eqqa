import json
import logging
from typing import Optional
from collections import defaultdict

from overrides import overrides

import torch

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import (
    MetadataField,
    TextField,
)
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer


logger = logging.getLogger(__name__)


@DatasetReader.register("squad_modeling")
class SquadModelingReader(DatasetReader):
    def __init__(
        self,
        gpt2_model_name: str = "gpt2",
        max_sequence_length: int = 1024,
        sequence_pair_separator: Optional[str] = "<|endoftext|>",
        **kwargs,
    ) -> None:
        super().__init__(
            manual_distributed_sharding=True,
            manual_multiprocess_sharding=True,
            **kwargs,
        )
        self._gpt2_model_name = gpt2_model_name
        self._tokenizer = PretrainedTransformerTokenizer(
            gpt2_model_name, add_special_tokens=False
        )

        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(gpt2_model_name)
        }
        self._max_sequence_length = max_sequence_length
        self._sequence_pair_separator = sequence_pair_separator
        self._stats = defaultdict(int)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading json file at %s", file_path)

        dataset = [json.loads(line) for line in open(file_path)]
        for datum in self.shard_iterable(dataset):
            yield self.text_to_instance(datum["context"], datum["question"], datum["id"])
 
        print("Dataset stats:\n")
        for key, value in self._stats.items():
            print(f"{key}: {value}")

    @overrides
    def text_to_instance(
            self,
            context: str,
            question: str,
            question_id: str
    ) -> Instance:
        tokenized_context = self._tokenizer.tokenize(context)
        tokenized_question = self._tokenizer.tokenize(question)
        allowed_context_length = (
                self._max_sequence_length
                - len(tokenized_question)
                - 1  # for separator
                - len(self._tokenizer.sequence_pair_start_tokens)
                - len(self._tokenizer.sequence_pair_end_tokens)
        )

        if len(tokenized_context) > allowed_context_length:
            self._stats["truncated contexts"] += 1
            tokenized_context = tokenized_context[:allowed_context_length]

        tokenized_input = (
            self._tokenizer.sequence_pair_start_tokens
            + tokenized_context
            + [Token(self._sequence_pair_separator)]
            + tokenized_question
            + self._tokenizer.sequence_pair_end_tokens
        )

        input_field = TextField(tokenized_input)
        metadata_field = MetadataField(
                {
                    "question": question,
                    "context": context,
                    "question_id": question_id
                }
        )
        return Instance(
                {
                    "context_and_question": input_field,
                    "metadata": metadata_field
                }
        )

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["context_and_question"].token_indexers = self._token_indexers
