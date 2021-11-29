import unittest
from textflint.generation.transformation.WSD.swap_target import SwapTarget
from textflint.input.component.sample.wsd_sample import WSDSample
from nltk.corpus import wordnet as wn

data = {
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
wsd_sample = WSDSample(data)
swap_target = SwapTarget()


class TestSwapTarget(unittest.TestCase):

    def test_transform(self):
        def wn_sensekey2synset(sensekey):
            r"""
            Convert sensekey to synset.
            :param str sensekey: sense key according to wordnet
            :return synset: synset including this sense key
            :return lemma: lemma extracted from this sense key
            """
            lemma = sensekey.split('%')[0]
            for synset in wn.synsets(lemma):
                for lemma in synset.lemmas():
                    if lemma.key() == sensekey:
                        return synset, lemma
            return None, None

        trans = swap_target.transform(wsd_sample, n=1, field='sentence')
        for i, trans_sample in enumerate(trans):
            trans_data = trans_sample.dump()
            ori_instance = data['instance']
            trans_instance = trans_data['instance']
            # check instance length
            self.assertTrue(len(ori_instance), len(trans_instance))
            # check whether in the same synset
            for ori_ins, trans_ins in zip(ori_instance, trans_instance):
                ori_key, ori_start, ori_end, ori_word, ori_sk = ori_ins
                trans_key, trans_start, trans_end, trans_word, trans_sk = trans_ins
                # check key
                self.assertTrue(ori_key, trans_key)
                # check sense key
                ori_synset, _ = wn_sensekey2synset(ori_key)
                trans_synset, _ = wn_sensekey2synset(trans_key)
                # check whether the new word and the original word are from the same synset
                self.assertTrue(ori_synset == trans_synset)


if __name__ == "__main__":
    unittest.main()
