import unittest

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UTCN import CnHomophones

sample = UTCnSample({
    'x': '我接受了她的礼物。',
    'y': 1,
})
trans_method = CnHomophones(get_pos=True)


class TestHomophones(unittest.TestCase):
    def test_transformation(self):
        special_sample = UTCnSample({'x': '', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        special_sample = UTCnSample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        # test if the item change
        change_sample = trans_method.transform(sample, n=3)
        self.assertEqual(3, len(change_sample))
        for s in change_sample:
            self.assertEqual(sample.get_tokens('x')[:1], s.get_tokens('x')[:1])
            self.assertEqual(sample.get_tokens('x')[-6:], s.get_tokens('x')[-6:])


if __name__ == "__main__":
    unittest.main()
