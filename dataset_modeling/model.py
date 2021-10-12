from typing import Optional, Any, Dict

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util
import torch
from transformers.models.gpt2 import GPT2LMHeadModel


@Model.register("rc_dataset_model")
class ReadingComprehensionDatasetModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        gpt2_model_name: Optional[str] = "gpt2",
        **kwargs
    ):
        super().__init__(vocab, **kwargs)
        self._gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

    def forward(
        self,
        context_and_question: TextFieldTensors,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, torch.Tensor]:
        input_ids = util.get_token_ids_from_text_field_tensors(context_and_question)
        attention_mask = util.get_text_field_mask(context_and_question)
        output = self._gpt2_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
                return_dict=True,
                use_cache=False
        )
        output_dict = {"loss": output["loss"]}
        if metadata:
            output_dict["question_ids"] = [datum["question_id"] for datum in metadata]

        return output_dict
