import unittest
from TextFlint.input_layer.dataset import Dataset
from TextFlint.generation_layer.generator.ner_generator import NERGenerator

sample1 = {'x': 'Amy lives in a city , which is called NYK .',
           'y': ['B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'O']}
sample2 = {'x': 'Jotion lives in Xian 105 kilometers away .',
           'y': ['B-PER', 'O', 'O', 'B-LOC', 'O', 'O', 'O', 'O']}
sample3 = {'x': 'China rejects Syrians call to boycott Chinese lamb .',
           'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']}
single_data_sample = [sample1]
data_samples = [sample1, sample2, sample3]
dataset = Dataset('NER')
single_dataset = Dataset('NER')
dataset.load(data_samples)
single_dataset.load(single_data_sample)
gene = NERGenerator()


class TestSpecialEntityTyposSwap(unittest.TestCase):

    def test_generate(self):
        # test task transformation
        transformation_methods = ["SwapEnt", "EntTypos"]
        gene = NERGenerator(transformation_methods=transformation_methods,
                            subpopulation_methods=[])

        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(3, len(original_samples))
            for index in range(len(original_samples)):
                for ori_entity, trans_entity in \
                        zip(original_samples[index].entities,
                            trans_rst[index].entities):
                    self.assertTrue(ori_entity['entity'] !=
                                    trans_entity['entity'])
                ori_samples = original_samples[index].entities_replace(
                    list(original_samples[index].entities)[::-1],
                    ["A"] * len(original_samples[index].entities))
                tran_rst = trans_rst[index].entities_replace(
                    list(trans_rst[index].entities)[::-1],
                    ["A"] * len(trans_rst[index].entities))
                self.assertEqual(ori_samples.get_words('text'),
                                 tran_rst.get_words('text'))

        gene = NERGenerator(transformation_methods=['ConcatSent'],
                            subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(2, len(original_samples))
            sentence1 = []
            for sent in [sample2, sample3]:
                sentence1 += sent['x'].split(' ')
            self.assertEqual(sentence1, trans_rst[1].get_words('text'))
            sentence2 = sample1['x'].split(' ') + sentence1
            self.assertEqual(sentence2, trans_rst[0].get_words('text'))

        # test wrong transformation_methods
        gene = NERGenerator(transformation_methods=["wrong_transform_method"],
                            subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = NERGenerator(transformation_methods=["AddSubtree"],
                            subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = NERGenerator(transformation_methods="OOV",
                            subpopulation_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        # test part of UT transformations
        gene = NERGenerator(transformation_methods=['WordCase'],
                            subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(3, len(original_samples))
            for index in range(len(original_samples)):
                for trans_word, ori_word in \
                        zip(trans_rst[index].get_words('text'),
                            original_samples[index].get_words('text')):
                    self.assertEqual(trans_word, ori_word.lower())

        gene = NERGenerator(transformation_methods=['SwapNum'],
                            subpopulation_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(1, len(original_samples))
            for index in range(len(original_samples)):
                for trans_word, ori_word in \
                        zip(trans_rst[index].get_words('text'),
                            original_samples[index].get_words('text')):
                    if ori_word.isdigit():
                        self.assertTrue(ori_word != trans_word)


if __name__ == "__main__":
    unittest.main()
