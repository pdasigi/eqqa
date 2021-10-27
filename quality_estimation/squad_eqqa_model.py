from typing import Dict, Any

from overrides import overrides

import torch
from torch.nn import MSELoss

from allennlp.common.file_utils import cached_path
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.models.archival import load_archive
from allennlp.modules import FeedForward
from allennlp.training.metrics import MeanAbsoluteError, Average
from allennlp_models.rc import TransformerQA


@Model.register("squad_eqqa")
class SquadEqqaModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        trained_squad_model: str,
        max_sequence_length: int=512,
        output_projector: FeedForward = None,
        regression_layer: FeedForward = None,
        f1_range_classifier: FeedForward = None,
        padding_token_id: int = 1,
        one_class_weight: float = 0.2,
        regression_weight: float = 0.5,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.transformer = None
        cached_model_path = cached_path(trained_squad_model)
        model_archive = load_archive(cached_model_path)
        model = model_archive.model
        self._squad_embedder = model._text_field_embedder
        for _, param in model.named_parameters():
            param.requires_grad = False
        if output_projector:
            self._output_projector = output_projector
        else:
            embed_size = self._squad_embedder._token_embedders["tokens"].transformer_model.config.hidden_size
            self._output_projector = torch.nn.Linear(
                embed_size, 1
            )

        if regression_layer:
            self._regression_layer = regression_layer
        else:
            self._regression_layer = torch.nn.Linear(
                max_sequence_length, 1
            )

        # Classifier predicts whether the F1 score is 1.0 or other.
        if f1_range_classifier:
            self.f1_range_classifier = f1_range_classifier
        else:
            self.f1_range_classifier = torch.nn.Linear(
                max_sequence_length, 2
            )

        self._mae_metric = MeanAbsoluteError()
        self._predicted_f1 = Average()
        self._range_classifier_loss_weights = [1 - one_class_weight, one_class_weight]
        self._regression_weight = regression_weight
        self._regression_loss_function = MSELoss()
        self._range_accuracy = Average()
        self._padding_token_id = padding_token_id

    def forward(
        self,
        question_and_context: TextFieldTensors,
        target_f1: torch.Tensor = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        # We always pad the input ids to max length in the reader, but since the mask is created when the inputs
        # are being batched, the mask will be all True. We fix that here.
        token_ids = question_and_context["tokens"]["token_ids"]
        mask = torch.tensor(token_ids != self._padding_token_id, device=token_ids.device)
        question_and_context["tokens"]["mask"] = mask
        encoded_input = self._squad_embedder(question_and_context)
        # (batch_size, max_sequence_length)
        regression_input = self._output_projector(encoded_input).squeeze(-1)
        # (batch_size, 2)
        f1_range_prediction = self.f1_range_classifier(regression_input)

        target_f1_range = (target_f1 == 1.0) * 1

        loss_weights = torch.tensor(self._range_classifier_loss_weights, device=f1_range_prediction.device)
        classifier_loss = torch.nn.functional.cross_entropy(
                f1_range_prediction,
                target_f1_range.squeeze(1),
                weight=loss_weights,
                reduction='none'
        ).mean()
        # (batch_size, 1)
        regression_predicted_f1 = torch.sigmoid(self._regression_layer(regression_input))
        non_one_predicted_f1 = (target_f1_range == 0.0) * regression_predicted_f1
        non_one_target_f1 = (target_f1_range == 0.0) * target_f1
        regression_loss = self._regression_loss_function(non_one_predicted_f1, non_one_target_f1)

        predicted_range = torch.argmax(f1_range_prediction, 1).unsqueeze(1)
        for instance_predicted_range, instance_target_range in zip(predicted_range, target_f1_range):
            self._range_accuracy(instance_predicted_range == instance_target_range)

        combined_prediction = ((predicted_range == 1.0) * 1.0) + ((predicted_range == 0.0) * regression_predicted_f1)

        self._mae_metric(combined_prediction, target_f1)
        for instance_predicted_f1 in combined_prediction:
            self._predicted_f1(instance_predicted_f1)

        loss = ((1 - self._regression_weight) * classifier_loss) + (self._regression_weight * regression_loss)

        return {
            "loss": loss,
            "predicted_f1": combined_prediction,
            "predicted_range": predicted_range,
            "target_f1": target_f1,
            "id": [d["question_id"] for d in metadata]
        }

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        metrics = self._mae_metric.get_metric(reset)
        metrics["predicted_f1"] = self._predicted_f1.get_metric(reset)
        metrics["range_accuracy"] = self._range_accuracy.get_metric(reset)
        return metrics
