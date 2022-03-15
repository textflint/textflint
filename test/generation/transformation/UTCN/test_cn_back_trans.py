import unittest

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UTCN import BackTrans


class TestBackTrans(unittest.TestCase):
    @unittest.skip("Manual test")
    def test_transformation(self):
        import random
        random.seed(100)
        sent1 = '那只敏捷的棕色狐狸跳过了那只懒惰的狗。'
        data_sample = UTCnSample({'x': sent1, 'y': "negative"})
        trans = BackTrans(device='cpu')
        x = trans.transform(data_sample, n=1)
        self.assertEqual('敏捷的棕色狐狸跳过懒狗', x[0].get_text('x'))



if __name__ == "__main__":
    unittest.main()
