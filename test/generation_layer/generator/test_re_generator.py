import unittest

from TextFlint.input_layer.dataset import Dataset
from TextFlint.generation_layer.generator.re_generator import REGenerator

sample1 = {'x': ["``", "The", "situation", "is", "very", "serious", ",", "''",
                 "Mattis", ",", "30", ",", "told", "reporters", "after",
                 "meeting", "with", "Ban", "in", "New", "York", "."],
               'subj': [8,8], 'obj': [10,10], 'y': 'age'}
sample2 = {'x': ['Ble', 'Goude', ',', 'born', 'in', '1972', 'in', 'Gbagbo',
                 ',', 'was', 'graduated', 'at', 'Fudan', 'University', '.'],
            'subj': [0,1], 'obj': [7,7], 'y': 'birth'}
sample3 = {'x': ["Heig", ",", "the", "director", "of", "Apple", ",", "was",
                 "graduated", "from", "Fudan", "University", "."],
            'subj': [0,0], 'obj': [5,5], 'y': 'employee'}

sample4 = {'x': ['', '', ''], 'subj':[0,0], 'obj':[0,0], 'y':'age' }
sample5= {'x': ['!', '@', '#', '$', '%', '&', '*', '(', ')'],
          'subj': [5,5], 'obj': [6,6], 'y': 'None'}

single_data_sample = [sample1]
data_samples = [sample1, sample2, sample3, sample4, sample5]
dataset = Dataset('RE')
single_dataset = Dataset('RE')
dataset.load(data_samples)
single_dataset.load(single_data_sample)


class TestSpecialEntityTyposSwap(unittest.TestCase):

    def test_generate(self):
        # test task transformation
        transformation_methods = ["SwapBirth", "SwapAge"]
        gene = REGenerator(transformation_methods=transformation_methods,
                           subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(1, len(original_samples))
            for index in range(len(original_samples)):
                self.assertTrue(original_samples[index] != trans_rst[index])

        # test wrong transformation_methods
        gene = REGenerator(transformation_methods=["wrong_transform_method"],
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = REGenerator(transformation_methods=["AddSubtree"],
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = REGenerator(transformation_methods="OOV",
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))


if __name__ == "__main__":
    unittest.main()
