import unittest

from textflint.input.dataset import Dataset
from textflint.generation.generator.wsd_generator import WSDGenerator
from textflint.common.settings import ALLOWED_TRANSFORMATIONS

sample1 = {
    'sentence': ['Your', 'Oct.', '6', 'editorial', '``', 'The', 'Ill',
                 'Homeless', '``', 'referred', 'to',
                 'research', 'by', 'us', 'and', 'six', 'of', 'our',
                 'colleagues', 'that', 'was', 'reported', 'in',
                 'the', 'Sept.', '8', 'issue', 'of', 'the', 'Journal', 'of',
                 'the', 'American', 'Medical',
                 'Association', '.'],
    'pos': ['PRON', 'NOUN', 'NUM', 'NOUN', '.', 'DET', 'NOUN', 'NOUN', '.',
            'VERB', 'PRT', 'NOUN', 'ADP', 'PRON',
            'CONJ', 'NUM', 'ADP', 'PRON', 'NOUN', 'DET', 'VERB', 'VERB',
            'ADP', 'DET', 'NOUN', 'NUM', 'NOUN', 'ADP',
            'DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'NOUN', 'NOUN', '.'],
    'lemma': ['you', 'oct.', '6', 'editorial', '``', 'the', 'ill',
              'homeless', '``', 'refer', 'to', 'research',
              'by', 'we', 'and', 'six', 'of', 'we', 'colleague', 'that',
              'be', 'report', 'in', 'the', 'sept.', '8',
              'issue', 'of', 'the', 'journal', 'of', 'the', 'american',
              'medical', 'association', '.'],
    'instance': [['d000.s000.t000', 9, 10, 'referred', 'refer%2:32:01::'],
                 ['d000.s000.t001', 11, 12, 'research',
                  'research%1:04:00::'],
                 ['d000.s000.t002', 21, 22, 'reported',
                  'report%2:32:04::']], 'sentence_id': 'd000.s000',
    'source': 'semeval2007'}
sample2 = {
    'sentence': ['In', 'a', 'recent', 'report', ',', 'the', 'Institute',
                 'of', 'Medicine', 'pointed', 'out', 'that',
                 'certain', 'health', 'problems', 'may', 'predispose', 'a',
                 'person', 'to', 'homelessness', ',',
                 'others', 'may', 'be', 'a', 'consequence', 'of', 'it', ',',
                 'and', 'a', 'third', 'category', 'is',
                 'composed', 'of', 'disorders', 'whose', 'treatment', 'is',
                 'difficult', 'or', 'impossible', 'if',
                 'a', 'person', 'lacks', 'adequate', 'shelter', '.'],
    'pos': ['ADP', 'DET', 'ADJ', 'NOUN', '.', 'DET', 'NOUN', 'ADP', 'NOUN',
            'VERB', 'VERB', 'ADP', 'ADJ', 'NOUN',
            'NOUN', 'VERB', 'VERB', 'DET', 'NOUN', 'PRT', 'NOUN', '.',
            'NOUN', 'VERB', 'VERB', 'DET', 'NOUN', 'ADP',
            'PRON', '.', 'CONJ', 'DET', 'ADJ', 'NOUN', 'VERB', 'VERB',
            'ADP', 'NOUN', 'PRON', 'NOUN', 'VERB', 'ADJ',
            'CONJ', 'ADJ', 'ADP', 'DET', 'NOUN', 'VERB', 'ADJ', 'NOUN',
            '.'],
    'lemma': ['in', 'a', 'recent', 'report', ',', 'the', 'institute', 'of',
              'medicine', 'point_out', 'point_out',
              'that', 'certain', 'health', 'problem', 'may', 'predispose',
              'a', 'person', 'to', 'homelessness', ',',
              'other', 'may', 'be', 'a', 'consequence', 'of', 'it', ',',
              'and', 'a', 'third', 'category', 'be',
              'compose', 'of', 'disorder', 'whose', 'treatment', 'be',
              'difficult', 'or', 'impossible', 'if', 'a',
              'person', 'lack', 'adequate', 'shelter', '.'],
    'instance': [
        ['d000.s009.t000', 9, 11, 'pointed out', 'point_out%2:32:01::'],
        ['d000.s009.t001', 14, 15, 'problems', 'problem%1:26:00::'],
        ['d000.s009.t002', 16, 17, 'predispose', 'predispose%2:31:00::'],
        ['d000.s009.t003', 18, 19, 'person', 'person%1:03:00::'],
        ['d000.s009.t004', 33, 34, 'category', 'category%1:14:00::'],
        ['d000.s009.t005', 35, 36, 'composed', 'compose%2:42:00::'],
        ['d000.s009.t006', 47, 48, 'lacks', 'lack%2:42:00::'],
        ['d000.s009.t007', 49, 50, 'shelter', 'shelter%1:26:00::']],
    'sentence_id': 'd000.s009',
    'source': 'semeval2007'}

single_data_sample = [sample1]
data_samples = [sample1, sample2,sample1, sample2,sample1]
dataset = Dataset('WSD')
single_dataset = Dataset('WSD')
dataset.load(data_samples)
single_dataset.load(single_data_sample)


class TestWSDGenerator(unittest.TestCase):

    def test_generate(self):
        # test task transformation
        gene = WSDGenerator(trans_methods=ALLOWED_TRANSFORMATIONS['WSD'],
                            sub_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            if original_samples is None:
                continue
            self.assertTrue(len(original_samples) == len(trans_rst))
            for index in range(len(original_samples)):
                self.assertTrue(original_samples[index] != trans_rst[index])

        # test wrong trans_methods

        gene = WSDGenerator(trans_methods=["wrong_transform_method"],
                            sub_methods=[])

        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = WSDGenerator(trans_methods=["AddSubtree"],
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = WSDGenerator(trans_methods="OOV",
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))

        # TODO:test pipeline trans_methods


if __name__ == "__main__":
    unittest.main()
