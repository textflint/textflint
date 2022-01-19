import unittest

from textflint.input.dataset import Dataset
from textflint.generation.generator.wsc_generator import WSCGenerator

sample1 = {"text": "The city councilmen refused the demonstrators a "
                   "permit because they feared violence.",
           "target": {"noun1": "The city councilmen", "noun2": "The demonstrators",
                      "noun1_idx": 0, "noun2_idx": 4, "pron": "they", "pron_idx": 9},
           "label": 0, "index": 0}
sample2 = {"text": "Tom threw his schoolbag down to Ray after he reached "
                   "the bottom of the stairs.",
           "target": {"noun1": "Tom", "noun2": "Ray",
                      "noun1_idx": 0, "noun2_idx": 6, "pron": "he", "pron_idx": 8},
           "label": 1, "index": 21}
sample3 = {"text": "John was doing research in the library when he heard a man "
                   "humming and  whistling. He was very annoyed.",
           "target": {"noun1": "John", "noun2": "The man",
                      "noun1_idx": 0, "noun2_idx": 10, "pron": "He", "pron_idx": 15},
           "label": 0, "index": 106}



single_data_sample = [sample1]
data_samples = [sample1, sample2, sample3]
dataset = Dataset(task='WSC')
single_dataset = Dataset(task='WSC')
dataset.load(data_samples)
single_dataset.load(single_data_sample)


class TestSpecialEntityTyposSwap(unittest.TestCase):

    def test_generate(self):

        # test UT transformation
        gene = WSCGenerator(trans_methods=['WordCase', 'Tense'],
                            sub_methods=[])

        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            for index in range(len(original_samples)):
                self.assertTrue(original_samples[index] != trans_rst[index])


        # test task transformation

        gene = WSCGenerator(trans_methods=['SwapNames', 'SwapGender'],
                            sub_methods=[])

        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            for index in range(len(original_samples)):
                self.assertTrue(original_samples[index] != trans_rst[index])


        # test wrong trans_methods
        gene = WSCGenerator(trans_methods=["wrong_transform_method"],
                           sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = WSCGenerator(trans_methods=["AddSubtree"],
                           sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = WSCGenerator(trans_methods="OOV",
                           sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))


if __name__ == "__main__":
    unittest.main()
