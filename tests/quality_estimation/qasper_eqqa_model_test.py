from allennlp.common.testing import ModelTestCase
import quality_estimation.qasper_eqqa_model
import quality_estimation.qasper_eqqa_reader

class TestQasperEqqaModel(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            "fixtures/experiment.json", "fixtures/data/qasper_sample_small.json"
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
