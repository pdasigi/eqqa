from typing import Dict, List, Tuple
from overrides import overrides
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import util
from allennlp.training.metrics import MeanAbsoluteError, PearsonCorrelation, Average
from torch.nn import MSELoss
from transformers import AutoModel, AutoTokenizer

import torch
import logging


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("mm_eqqa_roberta")
class MochaMetricModeling(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        target_metrics: List[str],
        target_correctness: str = None,
        transformer_model_name: str = "roberta-base",
        encoder_network: FeedForward = None,
        decoder_network: FeedForward = None,
        regression_layer: FeedForward = None,
        train_base: bool = True,
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self.transformer = AutoModel.from_pretrained(transformer_model_name)

        if not train_base:
            for _, param in self.transformer.named_parameters():
                param.requires_grad = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name, add_special_tokens=False
        )

        n_objectives = len(target_metrics)
        # ----------------------------------------------------------------
        # Define **ENCODER** head
        # ----------------------------------------------------------------
        if encoder_network:
            self.encoder_network = encoder_network
        else:
            self.encoder_network = torch.nn.Sequential([
                torch.nn.Linear(self.embedding_dim, 1),
                torch.nn.ReLU(inplace=True),
            ])

        # ----------------------------------------------------------------
        # Define **DECODER** head
        # ----------------------------------------------------------------
        if decoder_network:
            self.decoder_network = decoder_network
        else:
            self.decoder_network = torch.nn.Sequential([
                torch.nn.Linear(1, self.embedding_dim),
                torch.nn.ReLU(inplace=True),
            ])
        
        # ----------------------------------------------------------------
        # Define **Regression** head
        # ----------------------------------------------------------------
        if regression_layer:
            self.regression_layer = regression_layer
            self.n_objectives = regression_layer.get_output_dim()
        else:
            regr_inputs = self.decoder_network.get_input_dim()
            self.n_objectives = n_objectives

            self.regression_layer = torch.nn.Linear(
                regr_inputs, n_objectives
            )
        
        assert self.n_objectives == n_objectives, "Dimension mismatch!"
        self.regression_layer = torch.nn.Sequential(*self.regression_layer)

        # ----------------------------------------------------------------
        # Define **METRICS** head
        # ----------------------------------------------------------------
        self.target_metrics = target_metrics
        self.target_metrics2id = {m: i for i, m in enumerate(target_metrics)}
        self.target_correctness = target_correctness
        self._regression_losses = [MSELoss() for i in range(n_objectives)]

        self._mae_metric = MeanAbsoluteError()
        # self._pearson_metric =  PearsonCorrelation()

    @property
    def embedding_dim(self):
        return self.transformer.config.hidden_size

    def forward(
        self,
        input_text: TextFieldTensors,
        target_metrics: Tuple[str, torch.Tensor] = None,
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
        encoder_input = output['pooler_output']
        # ^Note: 
        # https://huggingface.co/docs/transformers/v4.18.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions.pooler_output
        encoder_output = self.encoder_network(encoder_input)
        decoder_output = self.decoder_network(encoder_output)      
        
        # (batch_size, n_objectives)
        prediction = self.regression_layer(decoder_output)
        #prediction = torch.sigmoid(prediction)

        output_dict = {
            "predicted_correctness": prediction,
        }

        if target_metrics is not None:
            loss_all_metrics = []
            for metric_name, metric_value in target_metrics:
                metric_id = self.target_metrics2id[metric_name]

                loss = self._regression_losses[metric_id]
                loss = loss(prediction[:, metric_id], metric_value)

                output_dict[f"loss_{metric_name}"] = loss
                output_dict[f"predicted_{metric_name}"] = prediction[:, metric_id]
                output_dict[f"target_{metric_name}"] = metric_value
                loss_all_metrics.append(loss)

            output_dict["avg_reg_loss"] = sum(loss_all_metrics) / len(loss_all_metrics)
            output_dict["max_reg_loss"] = max(loss_all_metrics)
        
        if target_correctness is not None:
            with torch.no_grad():
                encoder_output = torch.sigmoid(encoder_output.detach())
                output_dict["predicted_correctness"] = encoder_output
                output_dict["target_correctness"] = target_correctness
                self._mae_metric(encoder_output, target_correctness)
    
        return output_dict

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
       
        result = self._mae_metric.get_metric(reset)
        # result["pearson"] =  self._pearson_metric.get_metric(reset) 
        ## ^Note: apparently MAE returns a dictionary, whereas
        ## pearson correlation returns one value
        return result
