import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Iterable, Tuple
from torch import float16, float32

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
        target_correctness: str,
        target_metrics: List[str],
        transformer_model_name: str = "roberta-base",
        target_datasets: List[str] = "*",
        max_answer_length: int = 100,
        max_query_length: int = 100,
        max_document_length: int = 512,
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

        self.target_correctness = target_correctness
        self.target_metrics = target_metrics

        self.target_datasets = target_datasets
        self.include_context = include_context
        
        self.max_answer_length = max_answer_length
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length

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
            if self.is_included(dataset_name):
                yield from self._dataset_to_instances(dataset)
            
        logger.info("Stats:")
        for key, value in self._log_data.items():
            logger.info(f"{key}: {value}")

    def _dataset_to_instances(self, dataset: Dict[str, Any]) -> Iterable[Instance]:
        # Each example is composed of 
        # {
        #   "context": ...,
        #   "question": ...,
        #   "reference": ...,
        #   "candidate": ...,
        #   "metric_1": ..., 
        #   ..., 
        #   "metric_m": ..., 
        # }
        example = next(iter(dataset.items()))
        metrics = self.get_metrics(example)
        self._log_data["target_metrics"] = metrics.keys()

        for example_id, example in dataset.items():
            example_context = example["context"] if self.include_context else None
            target_metrics = self.get_metrics(example)

            yield self.text_to_instance(
                target_metrics=target_metrics,
                context=example_context,
                question=example["question"],
                candidate=example["candidate"],
                reference=example["reference"],
                target_correctness=example.get(self.target_correctness, None),
            )

    def _log_truncated(self, truncated_artifacts):
        truncated = []
        for param_name, is_truncated in truncated_artifacts.items():
            self._log_data[f"truncated_{param_name}"] += 1
            truncated.append(is_truncated)

        self._log_data[f"truncated_example"] += int(any(truncated))

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["input_text"].token_indexers = self._token_indexers

    @overrides
    def text_to_instance(
        self,  # type: ignore  # pylint: disable=arguments-differ
        question: str,
        candidate: str,
        reference: str,
        target_metrics: Tuple[str, float],
        target_correctness: float = None,
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
        input_text = []

        tokenized_can, truncated_can = _tokenize(candidate, self.max_answer_length)
        tokenized_ref, truncated_ref = _tokenize(reference, self.max_answer_length)
        
        input_text += tokenized_can
        input_text += [Token(self._paragraph_separator)] + tokenized_ref

        # Determine remaining length for question 
        # (this accounts for longer question-answer pairs)
        allowed_question_length = (
            self.max_document_length - len(input_text) - 1 # paragraph selector
        )

        allowed_question_length = min(self.max_query_length, allowed_question_length)
        tokenized_question, truncated_question = _tokenize(question, allowed_question_length)

        input_text += [Token(self._paragraph_separator)] + tokenized_question
        # Determine remaining length for context (if specified)
        allowed_context_length = (
                self.max_document_length - len(input_text) - 1  # for paragraph separator
        )

        use_context = context is not None
        tokenized_context, truncated_context = \
            _tokenize(context, allowed_context_length) if use_context else ([], False)

        input_text += [Token(self._paragraph_separator)] + tokenized_context
        self._log_truncated({
            "context": truncated_context, "question": truncated_question,
            "candidate": truncated_can,   "reference": truncated_ref,
        })

        fields["input_text"] = TextField(input_text)
        metrics, values = zip(*target_metrics) # FIXME - Handle NaNs
        fields["target_metrics_names"] = metrics
        fields["target_metrics_values"] = TensorField(torch.tensor(values, dtype=torch.float16))
        
        fields["target_correctness"] = self.get_correctness(target_correctness)
        return Instance(fields)

    def is_included(self, name):
        return self.target_datasets == "*" or (name in self.target_datasets)

    def get_metrics(self, example: Dict[str, any]) -> Tuple[str, float]:
        metrics = (m for m in sorted(example.keys()) if m in self.target_metrics)
        return Tuple((m, float32(example[m])) for m in metrics)

    def get_correctness(self, target):
        if target is None:
            return None

        if not (0 <= target <= 1):
            logger.warning(
                f"{self.target_correctness} should be in (0, 1) but is {target}")

        return TensorField(torch.tensor([target], dtype=torch.float16))

