import unittest

from textflint.input_layer.component.sample import SASample
from textflint.generation_layer.transformation.UT.back_trans import BackTrans


class TestBackTrans(unittest.TestCase):
    @unittest.skip("Manual test")
    def test_transformation(self):
        sent1 = 'To whom did the Virgin Mary allegedly appear in ' \
                '1858 in Lourdes France?'
        data_sample = SASample({'x': sent1, 'y': "negative"})
        trans = BackTrans(device='cpu')
        x = trans.transform(data_sample, n=1)
        self.assertEqual('To whom did the Virgin Mary allegedly appear '
                         'in Lourdes, France, in 1858?', x[0].get_text('x'))
        self.assertEqual('negative', x[0].get_value('y'))


if __name__ == "__main__":
    unittest.main()
