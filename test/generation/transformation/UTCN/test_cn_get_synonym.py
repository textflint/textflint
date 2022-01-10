import unittest

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UTCN import CnSynonym

sample = UTCnSample({
    'x': '我接受了她的礼物。',
    'y': 1,
})
trans_method = CnSynonym()


class TestSynonym(unittest.TestCase):
    def test_transformation(self):
        special_sample = UTCnSample({'x': '', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        special_sample = UTCnSample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        # test if the item change
        change_sample = trans_method.transform(sample, n=3)
        self.assertEqual(3, len(change_sample))


if __name__ == "__main__":
    unittest.main()
