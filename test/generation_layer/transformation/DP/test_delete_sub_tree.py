import unittest

from textflint.input_layer.component.sample.dp_sample import DPSample
from textflint.generation_layer.transformation.DP.delete_sub_tree \
    import DeleteSubTree

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
rms_ins = DeleteSubTree()


class TestSpecialRemoveSubtree(unittest.TestCase):
    def test_RemoveSubtree(self):
        ins_rms = rms_ins.transform(sample)[0]
        self.assertEqual(['Influential', 'members', 'of', 'the', 'House',
                          'Ways', 'and', 'Means', 'Committee', 'introduced',
                          'legislation', 'that', 'would', 'restrict', 'how',
                          'the', 'new', 'savings-and-loan', 'bailout',
                          'agency', 'can', 'raise', 'capital', '.'],
                         ins_rms.get_value('x'))

        change = rms_ins.transform(sample, n=5)
        self.assertTrue(5 >= len(change))

    def test_find_subtree(self):
        subtree = rms_ins.find_subtree(sample)
        self.assertEqual([(24, 37)], subtree)


if __name__ == '__main__':
    unittest.main()
