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
        regression_feedforward: FeedForward = None,
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
        if regression_feedforward:
            self.regression_feedforward = regression_feedforward
        else:
            self.regression_feedforward = torch.nn.Linear(self.transformer.config.hidden_size, 1)
        self.pooler = torch.nn.Linear(max_document_length, 1)
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
            decoder_input_ids=torch.tensor([[0, 1, 2]]),
            use_cache=False,
            return_dict=True,
            output_hidden_states=True
        )
        encoded_tokens = output["encoder_last_hidden_state"]
        answer_logits = output["logits"]
        prepooled_output = self.regression_feedforward(encoded_tokens)
        prediction = torch.sigmoid(self.pooler(prepooled_output.squeeze(-1)))
        loss = self._loss_function(prediction, target_f1)
        self._metric(prediction, target_f1)
        return {"loss": loss, "predicted_f1": prediction}

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        return self._metric.get_metric(reset)
