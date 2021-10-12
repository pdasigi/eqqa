# pylint: disable=no-self-use,invalid-name
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary
import numpy

from dataset_modeling.dataset_reader import SquadModelingReader


class TestQasperReader:
    def test_read_from_file(self):
        reader = SquadModelingReader()
        instances = ensure_list(reader.read("fixtures/data/squad_modeling_sample_small.json"))
        assert len(instances) == 5

        instance = instances[0]
        assert set(instance.fields.keys()) == {
            "context_and_question",
            "metadata",
        }

        token_text = [t.text for t in instance.fields["context_and_question"].tokens]
        assert len(token_text) == 180
        assert token_text[:15] == [
            'Muslims',
            'Ġhave',
            'Ġbeen',
            'Ġliving',
            'Ġin',
            'ĠTibet',
            'Ġsince',
            'Ġas',
            'Ġearly',
            'Ġas',
            'Ġthe',
            'Ġ8',
            'th',
            'Ġor',
            'Ġ9',
        ]

        assert token_text[-15:] == [
            'ĠChina',
            '.',
            '<|endoftext|>',
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
        ]

        metadata = instance.fields["metadata"].metadata
        assert metadata["question"] == "What group of people has been living in Tibet since 1959?"
        assert metadata["question_id"] == "5ad0097677cf76001a6867c0"
