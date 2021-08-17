import os
import tarfile
from typing import Dict
from overrides import overrides
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import util
from allennlp.training.metrics import MeanAbsoluteError
import torch
from torch.nn import MSELoss
from transformers import AutoModelForSeq2SeqLM


@Model.register("qasper_eqqa")
class QasperQualityEstimator(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        trained_qasper_model: str,
        max_document_length: int=15360,
        max_decoder_output_length: int=100,
        encoder_output_projector: FeedForward = None,
        decoder_output_projector: FeedForward = None,
        regression_layer: FeedForward = None,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.transformer = None
        cached_model_path = cached_path(trained_qasper_model)
        tarball_ptr = tarfile.open(cached_model_path)
        tarball_dir = os.path.dirname(cached_model_path)
        uncompressed_model_dir_name = tarball_ptr.getnames()[0]
        tarball_ptr.extractall(tarball_dir)
        pretrained_model_path = os.path.join(tarball_dir, uncompressed_model_dir_name)
        self.transformer = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_path)
        for _, param in self.transformer.named_parameters():
            param.requires_grad = False
        if encoder_output_projector:
            self.encoder_output_projector = encoder_output_projector
        else:
            self.encoder_output_projector = torch.nn.Linear(
                self.transformer.config.hidden_size, 1
            )

        if decoder_output_projector:
            self.decoder_output_projector = decoder_output_projector
        else:
            self.decoder_output_projector = torch.nn.Linear(
                self.transformer.config.hidden_size, 1
            )

        if regression_layer:
            self.regression_layer = regression_layer
        else:
            self.regression_layer = torch.nn.Linear(
                max_document_length + max_decoder_output_length, 1
            )

        self._max_decoder_output_length = max_decoder_output_length
        self._metric = MeanAbsoluteError()
        self._loss_function = MSELoss()

    def forward(
        self,
        question_with_context: TextFieldTensors,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor = None,
        target_f1: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = util.get_token_ids_from_text_field_tensors(question_with_context)

        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            decoder_input_ids=input_ids.new(torch.tensor([[0] * self._max_decoder_output_length])),
            use_cache=False,
            return_dict=True,
            output_hidden_states=True
        )
        encoded_tokens = output["encoder_last_hidden_state"]
        decoded_tokens = output["last_hidden_state"]
        # (batch_size, max_document_length)
        projected_encoder_output = self.encoder_output_projector(encoded_tokens).squeeze(-1)
        # (batch_size, max_decoder_output_length)
        projected_decoder_output = self.decoder_output_projector(decoded_tokens).squeeze(-1)
        regression_input = torch.cat(
            [projected_encoder_output, projected_decoder_output], -1
        )
        prediction = torch.sigmoid(self.regression_layer(regression_input))
        loss = self._loss_function(prediction, target_f1)
        self._metric(prediction, target_f1)
        return {"loss": loss, "predicted_f1": prediction}

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        return self._metric.get_metric(reset)
