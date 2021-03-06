from typing import Dict
from overrides import overrides
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import util
from allennlp.training.metrics import MeanAbsoluteError, PearsonCorrelation

import torch
from torch.nn import MSELoss
from transformers import AutoConfig, AutoModel, AutoTokenizer

import logging


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("mocha_eqqa_roberta")
class MochaQualityEstimator(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        transformer_model_name: str = "roberta-base",
        hidden_size: int = 1,
        train_base: bool = True,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.transformer = AutoModel.from_pretrained(transformer_model_name)

        # We do not want to train the base transformer models
        if not train_base:
            for _, param in self.transformer.named_parameters():
                param.requires_grad = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name,
            add_special_tokens=False
        )
        
        # ----------------------------------------------------------------
        # Define **Regression** head
        # ----------------------------------------------------------------
        self.regression_layer = [
            torch.nn.Linear(self.embedding_dim, hidden_size),
        ]
        if hidden_size > 1:
            self.regression_layer.extend([
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_size, 1),
                ])
        self.regression_layer = torch.nn.Sequential(*self.regression_layer)
        self.loss = MSELoss()

        # ----------------------------------------------------------------
        # Define **METRICS** head
        # ----------------------------------------------------------------
        self._mae_metric = MeanAbsoluteError()
        # self._pearson_metric =  PearsonCorrelation()

    @property
    def embedding_dim(self):
        return self.transformer.config.hidden_size

    def forward(
        self,
        input_text: TextFieldTensors,
        target_correctness: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:

        input_ids = util.get_token_ids_from_text_field_tensors(input_text)
        attention_mask = util.get_text_field_mask(input_text)

        output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # (batch_size, lf_hidden_size)
        regression_input = output['pooler_output']
        # ^Note: 
        # https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions.pooler_output
        
        # (batch_size, 1)
        prediction = self.regression_layer(regression_input)
        prediction = torch.sigmoid(prediction)

        output_dict = {
            "predicted_correctness": prediction,
        }

        if target_correctness is not None:
            loss = self.loss(prediction, target_correctness)
            output_dict["loss"] = loss
            output_dict["target_correctness"] = target_correctness

            self._mae_metric(prediction, target_correctness)
            # self._pearson_metric(prediction, target_correctness)

        return output_dict

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
       
        result = self._mae_metric.get_metric(reset)
        # result["pearson"] =  self._pearson_metric.get_metric(reset) 
        ## ^Note: apparently MAE returns a dictionary, whereas
        ## pearson correlation returns one value
        return result
