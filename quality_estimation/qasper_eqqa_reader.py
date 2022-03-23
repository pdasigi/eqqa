import json
import logging
import random
from enum import Enum
from collections import defaultdict
from typing import Any, Dict, List, Optional, Iterable, Tuple

from overrides import overrides

import spacy
import torch

from allennlp.common.util import JsonDict
from allennlp.data.fields import (
    MetadataField,
    TextField,
    IndexField,
    ListField,
    TensorField,
)
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import PretrainedTransformerIndexer
from allennlp.data.tokenizers import Token, PretrainedTransformerTokenizer


logger = logging.getLogger(__name__)


@DatasetReader.register("qasper_eqqa")
class QasperEqqaReader(DatasetReader):
    def __init__(
        self,
        transformer_model_name: str = "allenai/led-base-16384",
        max_query_length: int = 128,
        max_document_length: int = 15360,
        paragraph_separator: Optional[str] = "</s>",
        padding_token: Optional[str] = "<pad>",
        include_global_attention_mask: bool = True,
        target_f1_type: str = "max_f1",
        include_predictions: bool = False,
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

        self._include_global_attention_mask = include_global_attention_mask
        self._token_indexers = {
            "tokens": PretrainedTransformerIndexer(transformer_model_name)
        }
        self.max_query_length = max_query_length
        self.max_document_length = max_document_length
        self._paragraph_separator = paragraph_separator
        self._padding_token = padding_token
        assert target_f1_type in ["mean_f1", "max_f1"]
        self._target_f1_type = target_f1_type
        self._include_predictions = include_predictions
        self._log_data = defaultdict(int)

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading json file at %s", file_path)
        with open_compressed(file_path) as dataset_file:
            dataset = json.load(dataset_file)
        for article_id, article in self.shard_iterable(dataset.items()):
            if not article["full_text"]:
                continue
            article["article_id"] = article_id
            yield from self._article_to_instances(article)

        logger.info("Stats:")
        for key, value in self._log_data.items():
            logger.info(f"{key}: {value}")

    def _article_to_instances(self, article: Dict[str, Any]) -> Iterable[Instance]:
        paragraphs = self._get_paragraphs_from_article(article)
        tokenized_context = None
        tokenized_context, paragraph_start_indices = self._tokenize_paragraphs(
            paragraphs
        )

        for question_info in article["qas"]:
            model_prediction = question_info["model_prediction"] if self._include_predictions else None
            yield self.text_to_instance(
                question_info["question"],
                tokenized_context,
                paragraph_start_indices,
                question_info[self._target_f1_type],
                model_prediction
            )

    @overrides
    def text_to_instance(
        self,  # type: ignore  # pylint: disable=arguments-differ
        question: str,
        tokenized_context: List[Token],
        paragraph_start_indices: List[int],
        target_f1: float = None,
        model_prediction: str = None,
    ) -> Instance:
        fields = {}

        tokenized_question = self._tokenizer.tokenize(question)
        if len(tokenized_question) > self.max_query_length:
            tokenized_question = tokenized_question[:self.max_query_length]

        tokenized_prediction = self._tokenizer.tokenize(model_prediction) if model_prediction is not None else []
        allowed_context_length = (
                self.max_document_length
                - len(tokenized_question)
                - len(self._tokenizer.sequence_pair_start_tokens)
                - len(tokenized_prediction)
                - 2  # for paragraph seperators
        )

        if len(tokenized_context) > allowed_context_length:
            self._log_data["truncated instances"] += 1
        tokenized_context = tokenized_context[:allowed_context_length]

        question_and_context = (
            self._tokenizer.sequence_pair_start_tokens
            + tokenized_question
            + [Token(self._paragraph_separator)]
            + tokenized_prediction
            + [Token(self._paragraph_separator)]
            + tokenized_context
        )

        if len(question_and_context) < self.max_document_length:
            original_question_context_length = len(question_and_context)
            padding_length = self.max_document_length - original_question_context_length
            question_and_context = question_and_context + [Token(self._padding_token)] * padding_length
            attention_mask = ([True] * original_question_context_length) + ([False] * padding_length)
        else:
            attention_mask = [True for _ in question_and_context]

        # make the question field
        question_field = TextField(question_and_context)
        fields["question_with_context"] = question_field

        # make the attention mask field
        fields["attention_mask"] = TensorField(torch.tensor(attention_mask))

        start_of_context = (
            len(self._tokenizer.sequence_pair_start_tokens)
            + len(tokenized_question)
            + 1
            + len(tokenized_prediction)
        )

        paragraph_indices_list = [x + start_of_context for x in paragraph_start_indices]

        if self._include_global_attention_mask:
            # We need to make a global attention array. We'll use all the paragraph indices and the
            # indices of question tokens.
            mask_indices = set(list(range(start_of_context)) + paragraph_indices_list)
            mask = [
                True if i in mask_indices else False for i in range(len(question_field))
            ]
            fields["global_attention_mask"] = TensorField(torch.tensor(mask))

        if target_f1 is not None:
            fields["target_f1"] = TensorField(torch.tensor([target_f1]))

        return Instance(fields)

    @overrides
    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields["question_with_context"].token_indexers = self._token_indexers

    def _tokenize_paragraphs(
        self, paragraphs: List[str]
    ) -> Tuple[List[Token], List[int]]:
        tokenized_context = []
        paragraph_start_indices = []
        for paragraph in paragraphs:
            tokenized_paragraph = self._tokenizer.tokenize(paragraph)
            paragraph_start_indices.append(len(tokenized_context))
            tokenized_context.extend(tokenized_paragraph)
            if self._paragraph_separator:
                tokenized_context.append(Token(self._paragraph_separator))
        if self._paragraph_separator:
            # We added the separator after every paragraph, so we remove it after the last one.
            tokenized_context = tokenized_context[:-1]
        return tokenized_context, paragraph_start_indices

    def _get_paragraphs_from_article(self, article: JsonDict) -> List[str]:
        full_text = article["full_text"]
        paragraphs = []
        for section_info in full_text:
            # TODO (pradeep): It is possible there are other discrepancies between plain text, LaTeX and HTML.
            # Do a thorough investigation and add tests.
            if section_info["section_name"] is not None:
                paragraphs.append(section_info["section_name"])
            for paragraph in section_info["paragraphs"]:
                paragraph_text = paragraph.replace("\n", " ").strip()
                if paragraph_text:
                    paragraphs.append(paragraph_text)
        return paragraphs
