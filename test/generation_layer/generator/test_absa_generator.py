import unittest

from textflint.input_layer.dataset import Dataset
from textflint.generation_layer.generator.absa_generator import ABSAGenerator

sample1 = {"sentence": "BEST spicy tuna roll, great asian salad.", "term_list":
    {"35390182#756337#4_0": {"id": "35390182#756337#4_0",
                             "polarity": "positive", "term": "asian salad",
                             "from": 28, "to": 39, "opinion_words": ["great"],
                             "opinion_position": [[22, 27]]},
     "35390182#756337#4_1": {"id": "35390182#756337#4_1", "polarity":
         "positive", "term": "spicy tuna roll", "from": 5, "to": 20,
                             "opinion_words": ["BEST"],
                             "opinion_position": [[0, 4]]}}}
sample2 = {"sentence": "I love the drinks, esp lychee martini, "
                       "and the food is also VERY good.",
           "term_list": {"11447227#436718#3_0": {"id": "11447227#436718#3_0",
                                                 "polarity": "positive",
                                                 "term": "drinks",
                                                 "from": 11, "to": 17,
                                                 "opinion_words": ["love"],
                                                 "opinion_position": [[2, 6]]},
                         "11447227#436718#3_1": {"id": "11447227#436718#3_1",
                                                 "polarity": "positive",
                                                 "term": "lychee martini",
                                                 "from": 23, "to": 37,
                                                 "opinion_words": ["love"],
                                                 "opinion_position": [[2, 6]]},
                         "11447227#436718#3_2": {"id": "11447227#436718#3_2",
                                                 "polarity": "positive",
                                                 "term": "food", "from": 47,
                                                 "to": 51,
                                                 "opinion_words": ["good"],
                                               "opinion_position": [[65, 69]]}}}
sample3 = {"sentence": "Once we sailed, the top-notch food and "
                       "live entertainment sold us on a unforgettable evening.",
           "term_list": {"11313431#524365#3_1": {"id": "11313431#524365#3_1",
                                                 "polarity": "positive",
                                                 "term": "food", "from": 30,
                                                 "to": 34,
                                                 "opinion_words": ["top-notch"],
                                                "opinion_position": [[20, 29]]},
                         "11313431#524365#3_0": {"id": "11313431#524365#3_0",
                                                 "polarity": "positive", "term":
                                                     "live entertainment",
                                                 "from": 39, "to": 57,
                                                 "opinion_words": ["top-notch"],
                                               "opinion_position": [[20, 29]]}}}
sample4 = {"sentence": "                 ",
           "term_list": {"35390182#756337#4_0":
                             {"id": "35390182#756337#4_0",
                              "polarity": "positive", "term": " ",
                              "from": 3, "to": 4, "opinion_words": [" "],
                              "opinion_position": [[10, 11]]}}}
sample5 = {"sentence": "! @ # $ % ^ & * ( )", "term_list":
    {"35390182#756337#4_0": {"id": "35390182#756337#4_0",
                             "polarity": "positive", "term": "!",
                             "from": 0, "to": 1, "opinion_words": ["@"],
                             "opinion_position": [[2, 3]]}}}

data_samples = [sample1, sample2, sample3]
dataset = Dataset('ABSA')
dataset.load(data_samples)

special_samples = [sample4, sample5]
special_dataset = Dataset('ABSA')
special_dataset.load(special_samples)


class TestABSAGenerator(unittest.TestCase):

    def test_generate(self):
        # test task transformation
        transformation_methods = ['RevTgt', 'RevNon', 'AddDiff']
        gene = ABSAGenerator(transformation_methods=transformation_methods,
                             subpopulation_methods=[],
                             dataset_config='restaurant')

        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(3, len(original_samples))

        # test wrong transformation_methods
        gene = ABSAGenerator(transformation_methods=["wrong_transform_method"],
                             subpopulation_methods=[],
                             dataset_config='restaurant')
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = ABSAGenerator(transformation_methods=["Addabc"],
                             subpopulation_methods=[],
                             dataset_config='restaurant')
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = ABSAGenerator(transformation_methods=["AddSubtree"],
                             subpopulation_methods=[],
                             dataset_config='restaurant')
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = ABSAGenerator(transformation_methods="OOV",
                             subpopulation_methods=[],
                             dataset_config='restaurant')
        self.assertRaises(ValueError, next, gene.generate(dataset))

        # test pipeline transformation_methods
        transformation_methods = [['RevTgt', 'RevNon'],
                             ['RevNon', 'AddDiff']]
        gene = ABSAGenerator(transformation_methods=transformation_methods,
                             subpopulation_methods=[],
                             dataset_config='restaurant')
        for original_samples, trans_rst, trans_type \
                in gene.generate(special_dataset):
            self.assertEqual(len(trans_rst), len(original_samples))

        # test special data
        transformation_methods = ['AddDiff']
        gene = ABSAGenerator(transformation_methods=transformation_methods,
                             subpopulation_methods=[],
                             dataset_config='restaurant')
        for original_samples, trans_rst, trans_type \
                in gene.generate(special_dataset):
            self.assertEqual(len(trans_rst), len(original_samples))


if __name__ == "__main__":
    unittest.main()
