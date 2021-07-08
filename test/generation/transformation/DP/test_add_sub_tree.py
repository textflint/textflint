import unittest

from textflint.input.component.sample.dp_sample import DPSample
from textflint.generation.transformation.DP.add_sub_tree import AddSubTree
from textflint.common.utils.error import FlintError

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
ads_ins = AddSubTree()


class TestSpecialAddSubtree(unittest.TestCase):
    def test_AddSubtree(self):
        try:
            sample_trans = ads_ins.transform(sample)
        except FlintError:
            print('Cannot access Wikidata')
        else:
            if sample_trans:
                ins_ads = sample_trans[0]
                self.assertEqual(['Influential', 'members', 'of', 'the',
                          'House', 'Ways', 'and', 'Means', 'Committee', ',',
                          'introduced', 'legislation', 'that', 'would',
                          'restrict', 'how', 'the', 'new', 'savings-and-loan',
                          'bailout', 'agency', 'can', 'raise', 'capital', ',',
                          'creating', 'another', 'potential', 'obstacle',
                          'to', 'the', 'government', "'s", 'sale', 'of',
                          'sick', 'thrifts', '.'], ins_ads.get_value('x'))
            for trans in sample_trans:
                words = trans.get_value('x')
                self.assertTrue(',,' not in ''.join(words))

        data_no_entity = {'word': word, 'head': head, 'deprel': deprel,
                          'postag': ['NN'] * len(word)}
        try:
            sample_trans = ads_ins.transform(DPSample(data_no_entity))
        except FlintError:
            print('Cannot access Wikidata')
        else:
            self.assertEqual([], sample_trans)

        try:
            change = ads_ins.transform(sample, n=5)
        except FlintError:
            print('Cannot access Wikidata')
        else:
            self.assertTrue(5 >= len(change))

    def test_find_entity(self):
        entity_list = ads_ins.find_entity(sample)
        self.assertTrue(type(entity_list) == list)
        for i, word_id in enumerate(entity_list):
            self.assertTrue(type(word_id) == list)
            self.assertEqual(len(word_id), word_id[-1] - word_id[0] + 1)
            if i > 0:
                self.assertTrue(word_id[0] - entity_list[i - 1][-1] > 1)

    def test_get_clause(self):
        entity = 'House%20Ways%20and%20Means%20Committee'
        try:
            clause_add = ads_ins.get_clause(entity)
        except FlintError:
            print('Cannot access Wikidata')
        else:
            self.assertEqual('which is a Chief tax-writing comittee of the '
                             'United States House of Representatives',
                             clause_add)


if __name__ == '__main__':
    unittest.main()
