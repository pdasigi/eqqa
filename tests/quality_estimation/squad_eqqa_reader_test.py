# pylint: disable=no-self-use,invalid-name
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary
import numpy

from quality_estimation.squad_eqqa_reader import SquadEqqaReader


class TestSquadReader:
    def test_read_from_file(self):
        reader = SquadEqqaReader()
        instances = ensure_list(reader.read("fixtures/data/squad_modeling_sample_small.json"))
        assert len(instances) == 5

        instance = instances[0]
        assert set(instance.fields.keys()) == {
            "question_and_context",
            "target_f1",
            "metadata",
        }

        token_text = [t.text for t in instance.fields["question_and_context"].tokens]
        assert len(token_text) == 512
        assert token_text[:20] == [
            '<s>',
            'What',
            'Ġgroup',
            'Ġof',
            'Ġpeople',
            'Ġhas',
            'Ġbeen',
            'Ġliving',
            'Ġin',
            'ĠTibet',
            'Ġsince',
            'Ġ1959',
            '?',
            '</s>',
            '</s>',
            'Muslims',
            'Ġhave',
            'Ġbeen',
            'Ġliving',
            'Ġin',
        ]

        assert token_text[168:183] == [
            'Ġwhich',
            'Ġtraces',
            'Ġits',
            'Ġancestry',
            'Ġback',
            'Ġto',
            'Ġthe',
            'ĠH',
            'ui',
            'Ġethnic',
            'Ġgroup',
            'Ġof',
            'ĠChina',
            '.',
            '</s>'
        ]

        assert token_text[183:] == ['<pad>'] * (512 - 183)
        target_f1 = instance.fields["target_f1"].tensor.detach().numpy()
        numpy.testing.assert_almost_equal(target_f1, 1.0)
        metadata = instance.fields["metadata"].metadata
        assert metadata["question"] == "What group of people has been living in Tibet since 1959?"
        assert metadata["question_id"] == "5ad0097677cf76001a6867c0"

    def test_read_from_file_with_lower_ones_ratio(self):
        reader = SquadEqqaReader(ratio_of_ones_to_keep=0.55)
        instances = ensure_list(reader.read("fixtures/data/squad_modeling_sample_small.json"))
        assert len(instances) == 2
