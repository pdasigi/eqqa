import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from allennlp.common.file_util import cached_path
from allennlp.common.util import sanitize_wordpiece
from allennlp.data import Vocabulary
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.training.metrics import Average

from allennlp_models.rc.tools import squad

from calibration.util import get_k_best_spans

logger = logging.getLogger(__name__)


@Model.register("transformer_qa_calibrator")
class TransformerQACalibrator(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        serialized_model: str,
        num_samples: int = 20,
        mle_loss_weight: float = 1.0,
        ranking_loss_weight: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        archive = load_archive(cached_path(serialized_model))
        self._model = archive.model
        self._num_samples = num_samples
        self._mle_loss_weight = mle_loss_weight
        self._ranking_loss_weight = ranking_loss_weight
        self._margin_loss_function = nn.MarginRankingLoss()
        self._top_f1 = Average()
        self._oracle_f1 = Average()
        self._oracle_f1_rank = Average()
        self._ranking_loss = Average()

    def forward(  # type: ignore
        self,
        question_with_context: Dict[str, Dict[str, torch.LongTensor]],
        context_span: torch.IntTensor,
        cls_index: torch.LongTensor = None,
        answer_span: torch.IntTensor = None,
        metadata: List[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        model_outputs = self._model(
            question_with_context,
            context_span,
            cls_index,
            answer_span,
            metadata
        )
        k_best_spans = get_k_best_spans(model_outputs["span_start_logits"],
                                        model_outputs["span_end_logits"],
                                        self._num_samples)


        # (batch_size, k)
        best_span_probs = torch.gather(
            model_outputs["span_start_probs"], 1, k_best_spans[:, :, 0]
        ) * torch.gather(model_outputs["span_end_probs"], 1, k_best_spans[:, :, 1])

        mle_loss = model_outputs["loss"]

        k_prediction_strings = self._collect_best_span_strings(
            k_best_spans,
            context_span,
            metadata,
            cls_index
        )

        answers = [m["answers"] for m in metadata]
        f1_scores = []  # (batch_size * k)
        for instance_predictions, instance_answers in zip(k_prediction_strings, answers):
            instance_f1_scores = [max([squad.compute_f1(prediction, answer) for answer in instance_answers])
                                  for prediction in instance_predictions]
            self._top_f1(instance_f1_scores[0])
            max_f1 = max(instance_f1_scores)
            for i in range(len(instance_f1_scores)):
                if instance_f1_scores[i] == max_f1:
                    self._oracle_f1_rank(i+1)
            self._oracle_f1(max_f1)
            f1_scores.append(instance_f1_scores)

        targets_list = []
        input_scores_1 = []
        input_scores_2 = []
        for instance_f1_scores, instance_span_probs in zip(f1_scores, best_span_probs):
            targets_list.append([])
            input_scores_1.append([])
            input_scores_2.append([])
            for i in range(len(instance_f1_scores) - 1):
                for j in range(i+1, len(instance_f1_scores)):
                    input_scores_1[-1].append(instance_span_probs[i])
                    input_scores_2[-1].append(instance_span_probs[j])
                    targets_list[-1].append(1 if instance_f1_scores[i] >= instance_f1_scores[j] else -1)

        # (batch_size, kc2)
        scores_i = torch.tensor(input_scores_1)
        scores_j = torch.tensor(input_scores_2)
        target = torch.tensor(targets_list, device=scores_i.device)
        margin_loss = self._margin_loss_function(scores_i, scores_j, target)
        self._ranking_loss(margin_loss)
        loss = (self._mle_loss_weight * mle_loss) + (self._ranking_loss_weight * margin_loss)
        output_dict["loss"] = loss

        return output_dict

    def _collect_best_span_strings(
        self,
        k_best_spans: torch.Tensor,
        context_span: torch.IntTensor,
        metadata: List[Dict[str, Any]],
        cls_index: Optional[torch.LongTensor],
    ) -> Tuple[List[str], torch.Tensor]:
        _k_best_spans = k_best_spans.detach().cpu().numpy()

        best_span_strings: List[List[str]] = []

        for (metadata_entry, best_spans, cspan, cls_ind) in zip(
            metadata,
            _k_best_spans,
            context_span,
            cls_index or (0 for _ in range(len(metadata))),
        ):
            context_tokens_for_question = metadata_entry["context_tokens"]
            best_span_strings.append([])
            for best_span in best_spans:
                if best_span[0] == cls_ind:
                    # Predicting [CLS] is interpreted as predicting the question as unanswerable.
                    best_span_string = ""
                else:
                    best_span -= int(cspan[0])
                    assert np.all(best_span >= 0)

                    predicted_start, predicted_end = tuple(best_span)

                    while (
                        predicted_start >= 0
                        and context_tokens_for_question[predicted_start].idx is None
                    ):
                        predicted_start -= 1
                    if predicted_start < 0:
                        logger.warning(
                            f"Could not map the token '{context_tokens_for_question[best_span[0]].text}' at index "
                            f"'{best_span[0]}' to an offset in the original text."
                        )
                        character_start = 0
                    else:
                        character_start = context_tokens_for_question[predicted_start].idx

                    while (
                        predicted_end < len(context_tokens_for_question)
                        and context_tokens_for_question[predicted_end].idx is None
                    ):
                        predicted_end += 1
                    if predicted_end >= len(context_tokens_for_question):
                        logger.warning(
                            f"Could not map the token '{context_tokens_for_question[best_span[1]].text}' at index "
                            f"'{best_span[1]}' to an offset in the original text."
                        )
                        character_end = len(metadata_entry["context"])
                    else:
                        end_token = context_tokens_for_question[predicted_end]
                        character_end = end_token.idx + len(sanitize_wordpiece(end_token.text))

                    best_span_string = metadata_entry["context"][character_start:character_end]

                best_span_strings[-1].append(best_span_string)

        return best_span_strings

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        output = {
            "top_f1_score": self._top_f1.get_metric(reset),
            "oracle_f1_score": self._oracle_f1.get_metric(reset),
            "oracle_f1_rank": self._oracle_f1_rank.get_metric(reset),
            "ranking_loss": self._ranking_loss.get_metric(reset),
        }
        return output
