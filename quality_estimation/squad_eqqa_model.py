from typing import Dict, Any
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.models.archival import load_archive
from allennlp.modules import FeedForward
from allennlp.training.metrics import MeanAbsoluteError
from allennlp_models.rc import TransformerQA
import torch
from torch.nn import MSELoss, CrossEntropyLoss


@Model.register("squad_eqqa")
class SquadEqqaModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        trained_squad_model: str,
        max_sequence_length: int=512,
        output_projector: FeedForward = None,
        regression_layer: FeedForward = None,
        padding_token_id: int = 1,
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

        self._mae_metric = MeanAbsoluteError()
        self._regression_loss_function = MSELoss()
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
        # (batch_size, 1)
        predicted_f1 = torch.sigmoid(self._regression_layer(regression_input))
        loss = self._regression_loss_function(predicted_f1, target_f1)

        self._mae_metric(predicted_f1, target_f1)
        return {
            "loss": loss,
            "predicted_f1": predicted_f1,
            "target_f1": target_f1
        }

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        return self._mae_metric.get_metric(reset)
