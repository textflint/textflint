import unittest

from TextFlint.input_layer.dataset import Dataset
from TextFlint.generation_layer.generator.dp_generator import DPGenerator, sample, sample_1

single_data_sample = [sample]
data_samples = [sample, sample_1]
dataset = Dataset('DP')
dataset.load(data_samples)
gene = DPGenerator()


class TestDPGenerator(unittest.TestCase):

    def test_generate(self):
        # test task transformation
        gene = DPGenerator(transformation_methods=["DeleteSubTree"],
                           subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(2, len(original_samples))
            for original_sample, transformed_sample in \
                    zip(original_samples, trans_rst):
                self.assertTrue(len(original_sample.get_value('x')) !=
                                len(transformed_sample.get_value('x')))

        transformation_methods = ["DeleteSubTree", "Ocr"]
        gene = DPGenerator(transformation_methods=transformation_methods,
                           subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            for ori_sample, trans_sample in zip(original_samples, trans_rst):
                self.assertTrue(ori_sample != trans_sample)

        # test wrong transformation_methods
        gene = DPGenerator(transformation_methods=["wrong_transform_method"],
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = DPGenerator(transformation_methods=["EntityTyposSwap"],
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = DPGenerator(transformation_methods="RemoveSubtree",
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))

        # test part of UT transformations
        gene = DPGenerator(transformation_methods=['WordCase'],
                           subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(2, len(original_samples))
            for index in range(len(original_samples)):
                for trans_word, ori_word in \
                        zip(trans_rst[index].get_words('x'),
                            original_samples[index].get_words('x')):
                    self.assertEqual(trans_word, ori_word.lower())
        gene = DPGenerator(transformation_methods=['SwapNum'],
                           subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            for index in range(len(original_samples)):
                for trans_word, ori_word in \
                        zip(trans_rst[index].get_words('x'),
                            original_samples[index].get_words('x')):
                    if ori_word.isdigit():
                        self.assertTrue(ori_word != trans_word)


if __name__ == '__main__':
    unittest.main()
