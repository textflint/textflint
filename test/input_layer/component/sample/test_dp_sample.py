import unittest

from TextFlint.input_layer.component.sample.dp_sample import *

word = ['Influential', 'members', 'of', 'the', 'House', 'Ways', 'and',
        'Means', 'Committee', 'introduced', 'legislation', 'that',
        'would', 'restrict', 'how', 'the', 'new', 'savings-and-loan',
        'bailout', 'agency', 'can', 'raise', 'capital', ',', 'creating',
        'another', 'potential', 'obstacle', 'to', 'the', 'government',
        "'s", 'sale', 'of', 'sick', 'thrifts', '.']
postag = ['JJ', 'NNS', 'IN', 'DT', 'NNP', 'NNP', 'CC', 'NNP', 'NNP',
          'VBD', 'NN', 'WDT', 'MD', 'VB', 'WRB', 'DT', 'JJ', 'JJ', 'NN',
          'NN', 'MD', 'VB', 'NN', ',', 'VBG', 'DT', 'JJ', 'NN', 'TO',
          'DT', 'NN', 'POS', 'NN', 'IN', 'JJ', 'NNS', '.']
head = ['2', '10', '2', '6', '6', '3', '6', '9', '6', '0', '10', '14',
        '14', '11', '22', '20', '20', '20', '20', '22', '22', '14', '22',
        '14', '14', '28', '28', '25', '28', '31', '33', '31', '29', '33',
        '36', '34', '10']
deprel = ['amod', 'nsubj', 'prep', 'det', 'nn', 'pobj', 'cc', 'nn',
          'conj', 'root', 'dobj', 'nsubj', 'aux', 'rcmod', 'advmod',
          'det', 'amod', 'amod', 'nn', 'nsubj', 'aux', 'ccomp', 'dobj',
          'punct', 'xcomp', 'det', 'amod', 'dobj', 'prep', 'det', 'poss',
          'possessive', 'pobj', 'prep', 'amod', 'pobj', 'punct']

data = {'word': word, 'postag': postag, 'head': head, 'deprel': deprel}
sample = DPSample(data)


class TestDPSample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, DPSample, {'word': word})
        self.assertRaises(AssertionError, DPSample, {'postag': postag})
        self.assertRaises(AssertionError, DPSample, {'head': head})
        self.assertRaises(AssertionError, DPSample, {'deprel': deprel})
        self.assertRaises(AssertionError, DPSample,
                          {'word': word, 'postag': postag, 'head': head,
                           'deprel': 'amod'})
        self.assertRaises(IndexError, DPSample,
                          {'word': [], 'postag': [], 'head': [], 'deprel': []})
        postag_temp1 = postag[:-1]
        self.assertRaises(ValueError, DPSample,
                          {'word': word, 'postag': postag_temp1,
                           'head': head, 'deprel': deprel})
        postag_temp2 = postag + ['NN']
        self.assertRaises(ValueError, DPSample,
                          {'word': word, 'postag': postag_temp2,
                           'head': head, 'deprel': deprel})
        postag_temp3 = postag + ['']
        self.assertRaises(ValueError, DPSample,
                          {'word': word, 'postag': postag_temp3,
                           'head': head, 'deprel': deprel})
        word_temp = word + ['']
        self.assertRaises(ValueError, DPSample,
                          {'word': word_temp, 'postag': postag,
                           'head': head, 'deprel': deprel})
        deprel_temp = deprel[:-1] + ['nn']
        self.assertRaises(AssertionError, DPSample,
                          {'word': word, 'postag': postag,
                           'head': head, 'deprel': deprel_temp})

    def test_get_words(self):
        # test get words
        self.assertEqual(word, sample.get_words('x'))
        self.assertRaises(AttributeError, sample.get_words, 'word')
        self.assertRaises(AssertionError, sample.get_words, 'postag')

    def test_get_value(self):
        # test get value
        self.assertEqual(postag, sample.get_value('postag'))
        self.assertEqual(head, sample.get_value('head'))
        self.assertEqual(deprel, sample.get_value('deprel'))

    def test_dump(self):
        # test dump
        self.assertEqual({'word': word, 'postag': postag, 'head': head,
                          'deprel': deprel, 'sample_id': None},
                         sample.dump())

    def test_insert_field_after_index(self):
        # test insert after index
        ins_aft = sample.insert_field_after_index('x', 10, 'wug')

        self.assertRaises(IndexError, sample.insert_field_after_index,
                          'x', 100, 'wug')
        self.assertRaises(ValueError, sample.insert_field_after_index,
                          'x', -1, 'wug')
        self.assertRaises(AssertionError, sample.insert_field_after_index,
                          'word', 10, 'wug')

        self.assertEqual(['Influential', 'members', 'of', 'the', 'House',
                          'Ways', 'and', 'Means', 'Committee', 'introduced',
                          'legislation', 'wug', 'that', 'would', 'restrict',
                          'how', 'the', 'new', 'savings-and-loan', 'bailout',
                          'agency', 'can', 'raise', 'capital', ',',
                          'creating', 'another', 'potential', 'obstacle',
                          'to', 'the', 'government', "'s", 'sale', 'of',
                          'sick', 'thrifts', '.'], ins_aft.get_value('x'))
        self.assertEqual(['JJ', 'NNS', 'IN', 'DT', 'NNP', 'NNP', 'CC', 'NNP',
                          'NNP', 'VBD', 'NN', 'UNK', 'WDT', 'MD', 'VB', 'WRB',
                          'DT', 'JJ', 'JJ', 'NN', 'NN', 'MD', 'VB', 'NN', ',',
                          'VBG', 'DT', 'JJ', 'NN', 'TO', 'DT', 'NN', 'POS',
                          'NN', 'IN', 'JJ', 'NNS', '.'],
                         ins_aft.get_value('postag'))
        self.assertEqual(['2', '10', '2', '6', '6', '3', '6', '9', '6', '0',
                          '10', '0', '15', '15', '11', '23', '21', '21', '21',
                          '21', '23', '23', '15', '23', '15', '15', '29',
                          '29', '26', '29', '32', '34', '32', '30', '34',
                          '37', '35', '10'], ins_aft.get_value('head'))
        self.assertEqual(['amod', 'nsubj', 'prep', 'det', 'nn', 'pobj', 'cc',
                          'nn', 'conj', 'root', 'dobj', 'unk', 'nsubj', 'aux',
                          'rcmod', 'advmod', 'det', 'amod', 'amod', 'nn',
                          'nsubj', 'aux', 'ccomp', 'dobj', 'punct', 'xcomp',
                          'det', 'amod', 'dobj', 'prep', 'det', 'poss',
                          'possessive', 'pobj', 'prep', 'amod', 'pobj',
                          'punct'], ins_aft.get_value('deprel'))

    def test_insert_field_before_index(self):
        # test insert before index
        ins_bfr = sample.insert_field_before_index('x', 11, 'wug')

        self.assertRaises(IndexError, sample.insert_field_before_index,
                          'x', 100, 'wug')
        self.assertRaises(ValueError, sample.insert_field_before_index,
                          'x', -1, 'wug')
        self.assertRaises(AssertionError, sample.insert_field_before_index,
                          'word', 10, 'wug')

        self.assertEqual(['Influential', 'members', 'of', 'the', 'House',
                          'Ways', 'and', 'Means', 'Committee', 'introduced',
                          'legislation', 'wug', 'that', 'would', 'restrict',
                          'how', 'the', 'new', 'savings-and-loan', 'bailout',
                          'agency', 'can', 'raise', 'capital', ',',
                          'creating', 'another', 'potential', 'obstacle',
                          'to', 'the', 'government', "'s", 'sale', 'of',
                          'sick', 'thrifts', '.'], ins_bfr.get_value('x'))
        self.assertEqual(['JJ', 'NNS', 'IN', 'DT', 'NNP', 'NNP', 'CC', 'NNP',
                          'NNP', 'VBD', 'NN', 'UNK', 'WDT', 'MD', 'VB', 'WRB',
                          'DT', 'JJ', 'JJ', 'NN', 'NN', 'MD', 'VB', 'NN', ',',
                          'VBG', 'DT', 'JJ', 'NN', 'TO', 'DT', 'NN', 'POS',
                          'NN', 'IN', 'JJ', 'NNS', '.'],
                         ins_bfr.get_value('postag'))
        self.assertEqual(['2', '10', '2', '6', '6', '3', '6', '9', '6', '0',
                          '10', '0', '15', '15', '11', '23', '21', '21', '21',
                          '21', '23', '23', '15', '23', '15', '15', '29',
                          '29', '26', '29', '32', '34', '32', '30', '34',
                          '37', '35', '10'], ins_bfr.get_value('head'))
        self.assertEqual(['amod', 'nsubj', 'prep', 'det', 'nn', 'pobj', 'cc',
                          'nn', 'conj', 'root', 'dobj', 'unk', 'nsubj', 'aux',
                          'rcmod', 'advmod', 'det', 'amod', 'amod', 'nn',
                          'nsubj', 'aux', 'ccomp', 'dobj', 'punct', 'xcomp',
                          'det', 'amod', 'dobj', 'prep', 'det', 'poss',
                          'possessive', 'pobj', 'prep', 'amod', 'pobj',
                          'punct'], ins_bfr.get_value('deprel'))

    def test_delete_field_at_index(self):
        # test delete field at index
        ins_dlt = sample.delete_field_at_index('x', [10, 13])

        self.assertRaises(ValueError, sample.delete_field_at_index,
                          'x', [13, 10])
        self.assertRaises(ValueError, sample.delete_field_at_index,
                          'x', 100)
        self.assertRaises(ValueError, sample.delete_field_at_index,
                          'x', -1)
        self.assertRaises(AssertionError, sample.delete_field_at_index,
                          'word', 10)

        self.assertEqual(['Influential', 'members', 'of', 'the', 'House',
                          'Ways', 'and', 'Means', 'Committee', 'introduced',
                          'restrict', 'how', 'the', 'new', 'savings-and-loan',
                          'bailout', 'agency', 'can', 'raise', 'capital', ',',
                          'creating', 'another', 'potential', 'obstacle',
                          'to', 'the', 'government', "'s", 'sale', 'of',
                          'sick', 'thrifts', '.'], ins_dlt.get_value('x'))


if __name__ == '__main__':
    unittest.main()
