import unittest

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UTCN import CnDigit2Char

sample = UTCnSample({
    'x': '今天气温28摄氏度。',
    'y': 1,
})
trans_method = CnDigit2Char()


class TestDigit2Char(unittest.TestCase):
    def test_transformation(self):
        special_sample = UTCnSample({'x': '', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        special_sample = UTCnSample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        # test if the item change
        change_sample = trans_method.transform(sample)
        self.assertEqual(1, len(change_sample))
        for s in change_sample:
            self.assertEqual(sample.get_tokens('x')[:4], s.get_tokens('x')[:4])
            self.assertEqual(sample.get_tokens('x')[-4:], s.get_tokens('x')[-4:])
        self.assertEqual(change_sample[0].get_value('x'), '今天气温二十八摄氏度。')


if __name__ == "__main__":
    unittest.main()
