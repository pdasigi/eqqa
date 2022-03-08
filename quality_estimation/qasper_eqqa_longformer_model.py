from typing import Dict
from overrides import overrides
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import util
from allennlp.training.metrics import MeanAbsoluteError
import torch
from torch.nn import MSELoss
from transformers import AutoConfig, AutoModel, AutoTokenizer


@Model.register("qasper_eqqa_longformer")
class QasperQualityEstimator(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model_name: str = "allenai/longformer-base-4096",
        attention_dropout: float = 0.1,
        attention_window_size: int = 1024,
        gradient_checkpointing: bool = False,
        regression_layer: FeedForward = None,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        config = AutoConfig.from_pretrained(transformer_model_name)
        config.attention_dropout = attention_dropout
        config.attention_window = [attention_window_size] * len(config.attention_window)
        config.gradient_checkpointing = gradient_checkpointing
        self.transformer = AutoModel.from_pretrained(transformer_model_name, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name,
            add_special_tokens=False
        )

        if regression_layer:
            self.regression_layer = regression_layer
        else:
            self.regression_layer = torch.nn.Linear(
                config.hidden_size, 1
            )

        self._mae_metric = MeanAbsoluteError()
        self._regression_loss_function = MSELoss()

    def forward(
        self,
        question_with_context: TextFieldTensors,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor = None,
        target_f1: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = util.get_token_ids_from_text_field_tensors(question_with_context)
        attention_mask = util.get_text_field_mask(question_with_context)

        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            use_cache=False,
            return_dict=True
        )
        # (batch_size, lf_hidden_size)
        regression_input = output['pooler_output']

        # (batch_size, 1)
        prediction = torch.sigmoid(self.regression_layer(regression_input))
        loss = self._regression_loss_function(prediction, target_f1)

        self._mae_metric(prediction, target_f1)
        return {
            "loss": loss,
            "predicted_f1": prediction,
            "target_f1": target_f1
        }

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        metrics = self._mae_metric.get_metric(reset)
        return metrics
