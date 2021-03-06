import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Iterable

from overrides import overrides
import torch

from allennlp.data.fields import (
    TextField,
    TensorField,
)
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("mocha_eqqa")
class MochaEqqaReader(DatasetReader):
    def __init__(
        self,
        transformer_model_name: str = "roberta-base",
        max_answer_length: int = 100,
        max_query_length: int = 100,
        max_document_length: int = 512,
        exclude_datasets: Optional[List[str]] = None,
        include_context: bool = True,
        paragraph_separator: Optional[str] = "</s>",
        padding_token: Optional[str] = "<pad>",
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

        self.max_answer_length = max_answer_length
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length
        self.exclude_datasets = exclude_datasets

        self.include_context = include_context
        
        self._paragraph_separator = paragraph_separator
        self._padding_token = padding_token
        
        self._log_data = defaultdict(int)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading json file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            data = json.load(dataset_file)
        
        for dataset_name, dataset in self.shard_iterable(data.items()):
            # Skip dataset if user restricted the datasets to train on
            if (self.exclude_datasets is not None and dataset_name in self.exclude_datasets):
                continue

            yield from self._dataset_to_instances(dataset)

        logger.info("Stats:")
        for key, value in self._log_data.items():
            logger.info(f"{key}: {value}")


    def _dataset_to_instances(self, dataset: Dict[str, Any]) -> Iterable[Instance]:
        # Each dataset is composed of {example_id1: example, ...}        

        for example_id, example in dataset.items():
            example_context = example["context"] if self.include_context else None

            yield self.text_to_instance(
                context=example_context,
                question=example["question"],
                answer=example["answer"],
                target_correctness=example["correctness"],
            )

    @overrides
    def text_to_instance(
        self,  # type: ignore  # pylint: disable=arguments-differ
        question: str,
        answer: str,
        target_correctness: float,
        context: str = None,
    ) -> Instance:
        def _tokenize(text, max_length):
            truncated = False
            tokenized_text = self._tokenizer.tokenize(text)
            if len(tokenized_text) > max_length:
                tokenized_text = tokenized_text[:max_length]
                truncated=True
            return tokenized_text, truncated

        fields = {}

        tokenized_answer, _ = _tokenize(answer, self.max_answer_length)

        # Determine remaining length for question 
        # (this accounts for longer question-answer pairs)
        allowed_question_length = (
            self.max_document_length
            - len(tokenized_answer)
            - 1 # paragraph selector
        )
        allowed_question_length = min(self.max_query_length, allowed_question_length)
        tokenized_question, truncated_question = \
            _tokenize(question, allowed_question_length)

        # Determine remaining length for context (if specified)
        allowed_context_length = (
                self.max_document_length
                - len(tokenized_question)
                - len(tokenized_answer)
                - 2  # for paragraph separators
        )

        use_context = context is not None
        tokenized_context, truncated_context = \
            _tokenize(context, allowed_context_length) if use_context else []

        if truncated_question or (use_context and truncated_context):
            self._log_data["truncated instances"] += 1
            if truncated_question:
                self._log_data["truncated question"] += 1

        answer_question = (
            tokenized_answer
            + [Token(self._paragraph_separator)]
            + tokenized_question
            + [Token(self._paragraph_separator)]
            + tokenized_context
        )

        input_ids = TextField(answer_question)
        fields["input_text"] = input_ids

        # Apply min-max scaling
        if (target_correctness < 1) or (target_correctness > 5):
            logger.warning(f"Target correctness should be in the interval (1, 5) but {target_correctness}.")

        target = (target_correctness - 1) / (5-1)
        fields["target_correctness"] = TensorField(torch.tensor([target], dtype=torch.float16))
        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["input_text"].token_indexers = self._token_indexers
