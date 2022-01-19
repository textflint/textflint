import unittest

from textflint.input.component.sample import NMTSample
from textflint.generation.transformation.NMT.parallel_twitter_type\
    import ParallelTwitterType


class TestParallelTwitterType(unittest.TestCase):
    def test_transformation(self):
        source = "The last group on Friday evening departs at midnight, allowing riders 42 hours to complete the race."
        target = "Der letzten Freitags-Gruppe mit Startzeit um 24.00 Uhr stehen 42 Stunden zur Verfügung."
        data_sample = NMTSample({'source': source, 'target': target})
        swap_ins = ParallelTwitterType()
        x = swap_ins.transform(data_sample, n=5)

        self.assertEqual(5, len(x))
        change = []
        for sample in x:
            source_origin = data_sample.get_words('source')
            source_trans = sample.get_words('source')
            target_origin = data_sample.get_words('target')
            target_trans = sample.get_words('target')
            change.append(sample.get_text('source'))
            self.assertTrue(source_origin == source_trans[1:] or source_origin == source_trans[:-1])
            self.assertTrue(target_origin == target_trans[1:] or target_origin == target_trans[:-1])


if __name__ == "__main__":
    unittest.main()
