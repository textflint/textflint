import unittest

from textflint.input.component.sample import UTCnSample
from textflint.generation.transformation.UTCN \
    import CnSwapSynWordEmbedding


class TestCnSwapSynWordEmbedding(unittest.TestCase):
    def test_transformation(self):
        import random
        random.seed(1)
        sent1 = "那只敏捷的棕色狐狸跳过了那只懒惰的狗。"
        data_sample = UTCnSample({'x': sent1, 'y': "negative"})
        trans_method = CnSwapSynWordEmbedding()
        x = trans_method.transform(data_sample, n=5)
        self.assertTrue(5 == len(x))

        transformation_results = ['那只敏捷的棕色狐狸跳过了那只懒惰的故事。','那只敏捷的棕色狐狸跳过了那只懒惰的鱼。','那只敏捷的棕色狐狸跳过了那只懒惰的角色。','那只敏捷的棕色狐狸跳过了那只懒惰的人类。','那只敏捷的棕色狐狸跳过了那只懒惰的吃。',]


        for index, _sample in enumerate(x):
            self.assertTrue(transformation_results[index] == _sample.x.text)

        special_sample = UTCnSample({'x': '', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))
        special_sample = UTCnSample({'x': '~!@#$%^7890"\'', 'y': "negative"})
        self.assertEqual([], trans_method.transform(special_sample))


if __name__ == "__main__":
    unittest.main()
