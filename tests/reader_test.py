# pylint: disable=no-self-use,invalid-name
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary
import numpy

from qasper_eqqa.dataset_reader import QasperEqqaReader


class TestQasperReader:
    def test_read_from_file(self):
        reader = QasperEqqaReader()
        instances = ensure_list(reader.read("fixtures/data/qasper_sample_small.json"))
        assert len(instances) == 4

        instance = instances[1]
        assert set(instance.fields.keys()) == {
            "question_with_context",
            "global_attention_mask",
            "target_f1",
        }

        token_text = [t.text for t in instance.fields["question_with_context"].tokens]
        assert len(token_text) == 47
        assert token_text[:15] == [
            '<s>',
            'Are',
            'Ġthere',
            'Ġthree',
            '?',
            '</s>',
            'Introduction',
            '</s>',
            'A',
            'Ġshort',
            'Ġparagraph',
            '</s>',
            'Another',
            'Ġintro',
            'Ġparagraph'
        ]

        assert token_text[-15:] == [
            'ĠPol',
            'arity',
            'ĠFunction',
            '</s>',
            'Method',
            'Ġparagraph',
            'Ġusing',
            'Ġseed',
            'Ġlex',
            'icon',
            '</s>',
            'Conclusion',
            '</s>',
            'Conclusion',
            'Ġparagraph',
        ]

        numpy.testing.assert_almost_equal(instance["target_f1"].tensor.detach().numpy(), 0.9)
