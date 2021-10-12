from allennlp.common.testing import ModelTestCase
import dataset_modeling.model
import dataset_modeling.dataset_reader

class TestSquadModelingModel(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            "fixtures/dataset_modeling_experiment.json",
            "fixtures/data/squad_modeling_sample_small.json"
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
