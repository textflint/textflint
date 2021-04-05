import unittest
from TextFlint.input_layer.dataset import Dataset
from TextFlint.generation_layer.generator.sa_generator import SAGenerator

sample1 = {'x': 'Titanic is my favorite movie.',
           'y': 'pos'}
sample2 = {'x': 'I don\'t like the actor Tim Hill', 'y': 'neg'}
sample3 = {'x': 'The leading actor is good.',
           'y': 'pos'}
sample4 = {'x': '',
           'y': 'pos'}
sample5 = {'x': '!@#$$%^&*()_+}{|":?><',
           'y': 'pos'}
single_data_sample = [sample1]
data_samples = [sample1, sample2, sample3, sample4, sample5]
dataset = Dataset('SA')
single_dataset = Dataset('SA')
dataset.load(data_samples)
single_dataset.load(single_data_sample)


class TestSpecialEntityTyposSwap(unittest.TestCase):

    def test_generate(self):
        # test task transformation
        transformation_methods = ["SwapSpecialEnt", "AddSum",
                             "DoubleDenial", "SwapNum"]
        SA_config = {'AddSum': [{'entity_type': 'movie'},
                                {'entity_type': 'person'}],
                     'SwapSpecialEnt': [{'entity_type': 'movie'},
                                        {'entity_type': 'person'}]}
        gene = SAGenerator(transformation_methods=transformation_methods,
                           subpopulation_methods=[],
                           transformation_config=SA_config)

        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            for index in range(len(original_samples)):
                self.assertEqual(original_samples[index].y, trans_rst[index].y)

        # test wrong transformation_methods
        gene = SAGenerator(transformation_methods=["wrong_transform_method"],
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = SAGenerator(transformation_methods=["AddSubtree"],
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = SAGenerator(transformation_methods=["OOV"],
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = SAGenerator(transformation_methods="ReverseNeg",
                           subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))

        # test pipeline transformation_methods
        transformation_methods = [["SwapSpecialEnt", "AddSum"],
                             ["SwapSpecialEnt", "DoubleDenial"]]
        gene = SAGenerator(transformation_methods=transformation_methods,
                           subpopulation_methods=[],
                           transformation_config=SA_config)
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(len(trans_rst), len(original_samples))


if __name__ == "__main__":
    unittest.main()
