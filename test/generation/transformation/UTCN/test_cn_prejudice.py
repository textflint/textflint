import random
import unittest

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UTCN import CnPrejudice

sample = UTCnSample({
    'x': '李明这周去纽约。',
    'y': 1,
})


class TestPrejudice(unittest.TestCase):
    def test_transformation(self):
        trans_method = CnPrejudice(change_type='Name')
        special_sample = UTCnSample({'x': '', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        special_sample = UTCnSample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        random.seed(100)
        # test if the item change
        change_sample = trans_method.transform(sample, n=3)
        self.assertTrue(3 >= len(change_sample))
        # for s in change_sample:
        #     self.assertEqual(sample.get_tokens('x')[:1], s.get_tokens('x')[:1])
        #     self.assertEqual(sample.get_tokens('x')[-6:], s.get_tokens('x')[-6:])
        self.assertEqual(change_sample[0].get_value('x'), '韩雪冰这周去纽约。')

        trans_method = CnPrejudice(change_type='Loc', prejudice_tendency=['China'])
        special_sample = UTCnSample({'x': '', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        special_sample = UTCnSample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))

        random.seed(100)
        # test if the item change
        change_sample = trans_method.transform(sample, n=3)
        self.assertTrue(3 >= len(change_sample))
        # for s in change_sample:
        #     self.assertEqual(sample.get_tokens('x')[:1], s.get_tokens('x')[:1])
        #     self.assertEqual(sample.get_tokens('x')[-6:], s.get_tokens('x')[-6:])
        self.assertEqual(change_sample[-1].get_value('x'), '李明这周去银川。')


if __name__ == "__main__":
    unittest.main()
