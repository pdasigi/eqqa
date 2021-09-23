import os
import tarfile
from typing import Dict
from overrides import overrides
from allennlp.common.file_utils import cached_path, open_compressed
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import util
from allennlp.training.metrics import MeanAbsoluteError, Average
import torch
from torch.nn import MSELoss, CrossEntropyLoss
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
        f1_range_classifier: FeedForward = None,
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

        # Classifier predicts whether the F1 score is 0.0, 1.0, or other.
        if f1_range_classifier:
            self.f1_range_classifier = f1_range_classifier
        else:
            self.f1_range_classifier = torch.nn.Linear(
                max_document_length + max_decoder_output_length, 3
            )

        # Regression layer for predicting the F1 score that is "other".
        if regression_layer:
            self.regression_layer = regression_layer
        else:
            self.regression_layer = torch.nn.Linear(
                max_document_length + max_decoder_output_length, 1
            )

        self._max_decoder_output_length = max_decoder_output_length
        self._mae_metric = MeanAbsoluteError()
        # Check whether the predicted F1 is in the correct third.
        self._bucket_accuracy = Average()
        self._regression_loss_function = MSELoss()
        self._classifier_loss_function = CrossEntropyLoss()

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
            decoder_input_ids=torch.tensor([[0] * self._max_decoder_output_length]).to(input_ids.device),
            use_cache=False,
            return_dict=True,
            output_hidden_states=True
        )
        encoded_tokens = output["encoder_last_hidden_state"]
        decoded_tokens = output["decoder_hidden_states"][-1]
        # (batch_size, max_document_length)
        projected_encoder_output = self.encoder_output_projector(encoded_tokens).squeeze(-1)
        # (batch_size, max_decoder_output_length)
        projected_decoder_output = self.decoder_output_projector(decoded_tokens).squeeze(-1)
        # (batch_size, max_document_length + max_decoder_output_length)
        regression_input = torch.cat(
            [projected_encoder_output, projected_decoder_output], -1
        )
        # (batch_size, 3)
        f1_range_prediction = self.f1_range_classifier(regression_input)
        target_f1_range = self._make_f1_range_target(target_f1)
        classifier_loss = self._classifier_loss_function(f1_range_prediction, target_f1_range)
        # (batch_size, 1)
        other_prediction = torch.sigmoid(self.regression_layer(regression_input))
        # We should compute the regression loss only if the target f1 is neither 0.0 or 1.0.
        between_targets = (target_f1_range == 1.0) * target_f1
        between_predictions = (target_f1_range == 1.0) * other_prediction
        regression_loss = self._regression_loss_function(between_predictions, between_targets)
        loss = regression_loss + classifier_loss

        # Making the final prediction by combining the range and between predictions
        predicted_range = torch.argmax(f1_range_prediction, 1)
        # When the predicted range is neither class-1 or class-2, the prediction is 0.0
        prediction = ((predicted_range == 1.0) * between_predictions) + ((predicted_range == 2.0) * 1.0)

        self._mae_metric(prediction, target_f1)
        self._bucket_accuracy(self._get_bucket_accuracy(prediction, target_f1))
        return {
            "loss": loss,
            "predicted_f1": prediction,
            "predicted_range": predicted_range,
            "target_f1": target_f1
        }

    @staticmethod
    def _make_f1_range_target(target_f1: torch.Tensor) -> torch.Tensor:
        """
        Takes a tensor of F1 scores and makes a 3-class tensor indicating whether the f1 score is 0, between 0 and
        1, or is 1.
        """
        score_is_zero = target_f1 == 0.0
        score_is_one = target_score == 1.0
        score_is_between = (target_f1 > 0.0) * (target_f1 < 1.0)
        return ((score_is_zero * 1) + (score_is_between * 2) + (score_is_one * 3)) - 1

    @overrides
    def get_metrics(self, reset: bool=False) -> Dict[str, float]:
        metrics = self._mae_metric.get_metric(reset)
        metrics["bucket_accuracy"] = self._bucket_accuracy.get_metric(reset)
        return metrics

    @staticmethod
    def _get_bucket_accuracy(prediction: torch.Tensor, target_f1: torch.Tensor) -> float:
        bucket_accuracies = []
        for instance_prediction, instance_target in zip(prediction.tolist(),
                                                        target_f1.tolist()):
            prediction_bucket = None
            target_bucket = None
            for i, limit in enumerate([0.0, 0.134, 1.0]):
                if instance_prediction[0] <= limit and prediction_bucket is None:
                    prediction_bucket = i
                if instance_target[0] <= limit and target_bucket is None:
                    target_bucket = i

            bucket_accuracies.append(prediction_bucket == target_bucket)
        return sum(bucket_accuracies) / len(bucket_accuracies)
