from textflint.input.component.sample.coref_sample import *
import unittest
import random
from test.data.coref_debug import CorefDebug

sample1 = CorefDebug.coref_sample1()
sample2 = CorefDebug.coref_sample2()
sample3 = CorefDebug.coref_sample3()
sample4 = CorefDebug.coref_sample4()
sample5 = CorefDebug.coref_sample5()
sample6 = CorefDebug.coref_sample6()
samples = [sample1, sample2, sample3, sample4, sample5, sample6]

class TestCorefSample(unittest.TestCase):

    def test_sens2doc(self):
        for sample in samples:
            sens = sample.dump()["sentences"]
            doc, sen_map = CorefSample.sens2doc(sens)
            self.assertEqual(doc, sample.x.words)
            self.assertEqual(sen_map, sample.sen_map)
            sens1 = CorefSample.doc2sens(doc, sen_map)
            self.assertEqual(sens1, sens)

    def test_kth_sen(self):
        for sample in samples:
            sens = sample.dump()["sentences"]
            self.assertEqual(
                [sample.get_kth_sen(i) for i in range(sample.num_sentences())], 
                sens)

    def test_index_in_sen(self):
        self.assertEqual(
            sample1.eqlen_sen_map(), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        self.assertEqual(sample1.index_in_sen(5), 0)
        self.assertEqual(sample1.index_in_sen(6), 1)

    def test_concat_conlls(self):
        for i in range(5):
            concat_num = random.randint(2, 4)
            sample_ids = [random.randint(0, len(samples)-1)
                          for j in range(concat_num)]
            args = [samples[j] for j in sample_ids]
            sample_concat = CorefSample.concat_conlls(*args)
            curr_sennum = 0
            for j in range(concat_num):
                sample = samples[sample_ids[j]]
                for k in range(sample.num_sentences()):
                    k_from_s = sample.get_kth_sen(k)
                    k_from_sc = sample_concat.get_kth_sen(curr_sennum+k)
                    self.assertEqual(k_from_s, k_from_sc)
                curr_sennum += sample.num_sentences()
            
    def test_shuffle_conlls(self):
        a11 = CorefSample.concat_conlls(sample1, sample2)
        a13 = a11.shuffle_conll([0, 2, 1, 3])
        self.assertEqual(a13.dump(), {'doc_key': 'doc_key1', 'sentences':
            [['I', 'love', 'my', 'pet', 'Anna', '.'],
             ['Bob', "'s", 'wife', 'Anna', 'likes', 'winter', '.'],
             ['She', 'is', 'my', 'favorite', '.'],
             ['However', ',', 'he', 'loves', 'summer', '.']], 'speakers':
            [['sp1', 'sp1', 'sp1', 'sp1', 'sp1', 'sp1'],
             ['sp2', 'sp2', 'sp2', 'sp2', 'sp2', 'sp2', 'sp2'],
             ['sp1', 'sp1', 'sp1', 'sp1', 'sp1'],
             ['sp2', 'sp2', 'sp2', 'sp2', 'sp2', 'sp2']], 'clusters':
            [[[2, 3], [4, 4], [13, 13]], [[6, 8], [9, 9]], [[6, 6], [20, 20]]],
            'constituents': [[0, 0, '0-I-C'], [1, 1, '1-love-C'],
                             [2, 2, '2-my-C'], [3, 3, '3-pet-C'],
                             [4, 4, '4-Anna-C'], [5, 5, '5-.-C'],
                             [13, 13, '6-She-C'], [14, 14, '7-is-C'],
                             [15, 15, '8-my-C'], [16, 16, '9-favorite-C'],
                             [17, 17, '10-.-C'], [6, 6, '0-Bob-C'],
                             [7, 7, "1-'s-C"], [8, 8, '2-wife-C'],
                             [9, 9, '3-Anna-C'], [10, 10, '4-likes-C'],
                             [11, 11, '5-winter-C'], [12, 12, '6-.-C'],
                             [18, 18, '7-However-C'], [19, 19, '8-,-C'],
                             [20, 20, '9-he-C'], [21, 21, '10-loves-C'],
                             [22, 22, '11-summer-C'], [23, 23, '12-.-C']],
                                      'ner': [[0, 0, '0-I-N'], [1, 1, '1-love-N'],
                            [2, 2, '2-my-N'], [3, 3, '3-pet-N'],
                            [4, 4, '4-Anna-N'], [5, 5, '5-.-N'],
                            [13, 13, '6-She-N'], [14, 14, '7-is-N'],
                            [15, 15, '8-my-N'], [16, 16, '9-favorite-N'],
                            [17, 17, '10-.-N'], [6, 6, '0-Bob-N'],
                            [7, 7, "1-'s-N"], [8, 8, '2-wife-N'],
                            [9, 9, '3-Anna-N'], [10, 10, '4-likes-N'],
                            [11, 11, '5-winter-N'], [12, 12, '6-.-N'],
                            [18, 18, '7-However-N'], [19, 19, '8-,-N'],
                            [20, 20, '9-he-N'], [21, 21, '10-loves-N'],
                            [22, 22, '11-summer-N'], [23, 23, '12-.-N']],
                                      'sample_id': None})

    def test_part_conll(self):
        a11 = CorefSample.concat_conlls(sample1, sample2)
        a21 = a11.part_conll([0, 2])
        a22 = a11.part_conll([0, 1])
        self.assertEqual(
            a21.dump(),
            {'doc_key': 'doc_key1', 'sentences':
                [['I', 'love', 'my', 'pet', 'Anna', '.'],
                 ['Bob', "'s", 'wife', 'Anna', 'likes', 'winter', '.']],
             'speakers': [['sp1', 'sp1', 'sp1', 'sp1', 'sp1', 'sp1'],
                          ['sp2', 'sp2', 'sp2', 'sp2', 'sp2', 'sp2', 'sp2']],
             'clusters': [[[2, 3], [4, 4]], [[6, 8], [9, 9]], [[6, 6]]],
             'constituents': [[0, 0, '0-I-C'], [1, 1, '1-love-C'],
                              [2, 2, '2-my-C'], [3, 3, '3-pet-C'],
                              [4, 4, '4-Anna-C'], [5, 5, '5-.-C'],
                              [6, 6, '0-Bob-C'], [7, 7, "1-'s-C"],
                              [8, 8, '2-wife-C'], [9, 9, '3-Anna-C'],
                              [10, 10, '4-likes-C'], [11, 11, '5-winter-C'],
                              [12, 12, '6-.-C']],
             'ner': [[0, 0, '0-I-N'], [1, 1, '1-love-N'], [2, 2, '2-my-N'],
                     [3, 3, '3-pet-N'], [4, 4, '4-Anna-N'], [5, 5, '5-.-N'],
                     [6, 6, '0-Bob-N'], [7, 7, "1-'s-N"], [8, 8, '2-wife-N'],
                     [9, 9, '3-Anna-N'], [10, 10, '4-likes-N'],
                     [11, 11, '5-winter-N'], [12, 12, '6-.-N']],
             'sample_id': None})
        self.assertEqual(
            a22.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['I', 'love', 'my', 'pet', 'Anna', '.'],
                           ['She', 'is', 'my', 'favorite', '.']],
             'speakers': [['sp1', 'sp1', 'sp1', 'sp1', 'sp1', 'sp1'],
                          ['sp1', 'sp1', 'sp1', 'sp1', 'sp1']],
             'clusters': [[[2, 3], [4, 4], [6, 6]], [], []],
             'constituents': [[0, 0, '0-I-C'], [1, 1, '1-love-C'],
                              [2, 2, '2-my-C'], [3, 3, '3-pet-C'],
                              [4, 4, '4-Anna-C'], [5, 5, '5-.-C'],
                              [6, 6, '6-She-C'], [7, 7, '7-is-C'],
                              [8, 8, '8-my-C'], [9, 9, '9-favorite-C'],
                              [10, 10, '10-.-C']],
             'ner': [[0, 0, '0-I-N'],  [1, 1, '1-love-N'],  [2, 2, '2-my-N'],
                     [3, 3, '3-pet-N'], [4, 4, '4-Anna-N'], [5, 5, '5-.-N'],
                     [6, 6, '6-She-N'], [7, 7, '7-is-N'], [8, 8, '8-my-N'],
                     [9, 9, '9-favorite-N'], [10, 10, '10-.-N']],
             'sample_id': None})

    def test_part_before_conll(self):
        a11 = CorefSample.concat_conlls(sample1, sample2)
        a31 = a11.part_before_conll(2)
        a32 = a11.part_after_conll(2)
        self.assertEqual(
            a31.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['I', 'love', 'my', 'pet', 'Anna', '.'],
                           ['She', 'is', 'my', 'favorite', '.']],
             'speakers': [['sp1', 'sp1', 'sp1', 'sp1', 'sp1', 'sp1'],
                          ['sp1', 'sp1', 'sp1', 'sp1', 'sp1']],
             'clusters': [[[2, 3], [4, 4], [6, 6]], [], []],
             'constituents': [[0, 0, '0-I-C'], [1, 1, '1-love-C'],
                              [2, 2, '2-my-C'], [3, 3, '3-pet-C'],
                              [4, 4, '4-Anna-C'], [5, 5, '5-.-C'],
                              [6, 6, '6-She-C'], [7, 7, '7-is-C'],
                              [8, 8, '8-my-C'], [9, 9, '9-favorite-C'],
                              [10, 10, '10-.-C']],
             'ner': [[0, 0, '0-I-N'], [1, 1, '1-love-N'], [2, 2, '2-my-N'],
                     [3, 3, '3-pet-N'], [4, 4, '4-Anna-N'], [5, 5, '5-.-N'],
                     [6, 6, '6-She-N'], [7, 7, '7-is-N'], [8, 8, '8-my-N'],
                     [9, 9, '9-favorite-N'], [10, 10, '10-.-N']],
             'sample_id': None})
        self.assertEqual(
            a32.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['Bob', "'s", 'wife', 'Anna', 'likes', 'winter', '.'],
                           ['However', ',', 'he', 'loves', 'summer', '.']],
             'speakers': [['sp2', 'sp2', 'sp2', 'sp2', 'sp2', 'sp2', 'sp2'],
                          ['sp2', 'sp2', 'sp2', 'sp2', 'sp2', 'sp2']],
             'clusters': [[], [[0, 2], [3, 3]], [[0, 0], [9, 9]]],
             'constituents': [[0, 0, '0-Bob-C'], [1, 1, "1-'s-C"],
                              [2, 2, '2-wife-C'], [3, 3, '3-Anna-C'],
                              [4, 4, '4-likes-C'], [5, 5, '5-winter-C'],
                              [6, 6, '6-.-C'], [7, 7, '7-However-C'],
                              [8, 8, '8-,-C'], [9, 9, '9-he-C'],
                              [10, 10, '10-loves-C'], [11, 11, '11-summer-C'],
                              [12, 12, '12-.-C']],
             'ner': [[0, 0, '0-Bob-N'], [1, 1, "1-'s-N"], [2, 2, '2-wife-N'],
                     [3, 3, '3-Anna-N'], [4, 4, '4-likes-N'],
                     [5, 5, '5-winter-N'], [6, 6, '6-.-N'],
                     [7, 7, '7-However-N'], [8, 8, '8-,-N'], [9, 9, '9-he-N'],
                     [10, 10, '10-loves-N'], [11, 11, '11-summer-N'],
                     [12, 12, '12-.-N']], 'sample_id': None})
        
    def test_insert(self):
        a1 = sample1
        a61 = a1.insert_field_before_indices('x', [0], ['test'])
        a62 = a1.insert_field_before_indices(
            'x', [0, 2, 4], ['Wang', 'Xiao', ['fdu', 'jiangwan', 'cross_2']])
        a63 = a1.insert_field_after_indices(
            'x', [len(a1.x)-1], [[' haha', 'test']])
        a64 = a1.insert_field_after_indices(
            'x', [0, 2, 7], ['Wang', 'Xiao', ['fdu', 'jiangwan', 'cross_2']])
        self.assertEqual(
            a61.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['test', 'I', 'love', 'my', 'pet', 'Anna', '.'],
                           ['She', 'is', 'my', 'favorite', '.']],
             'speakers': [['sp_ins', 'sp1', 'sp1', 'sp1', 'sp1', 'sp1', 'sp1'],
                          ['sp1', 'sp1', 'sp1', 'sp1', 'sp1']],
             'clusters': [[[3, 4], [5, 5], [7, 7]]],
             'constituents': [[1, 1, '0-I-C'], [2, 2, '1-love-C'],
                              [3, 3, '2-my-C'], [4, 4, '3-pet-C'],
                              [5, 5, '4-Anna-C'], [6, 6, '5-.-C'],
                              [7, 7, '6-She-C'], [8, 8, '7-is-C'],
                              [9, 9, '8-my-C'], [10, 10, '9-favorite-C'],
                              [11, 11, '10-.-C']],
             'ner': [[1, 1, '0-I-N'], [2, 2, '1-love-N'], [3, 3, '2-my-N'],
                     [4, 4, '3-pet-N'], [5, 5, '4-Anna-N'], [6, 6, '5-.-N'],
                     [7, 7, '6-She-N'], [8, 8, '7-is-N'], [9, 9, '8-my-N'],
                     [10, 10, '9-favorite-N'], [11, 11, '10-.-N']],
             'sample_id': None})
        self.assertEqual(
            a62.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['Wang', 'I', 'love', 'Xiao', 'my', 'pet', 'fdu',
                            'jiangwan', 'cross_2', 'Anna', '.'],
                           ['She', 'is', 'my', 'favorite', '.']],
             'speakers': [['sp_ins', 'sp1', 'sp1', 'sp_ins', 'sp1', 'sp1',
                           'sp_ins', 'sp_ins', 'sp_ins', 'sp1', 'sp1'],
                          ['sp1', 'sp1', 'sp1', 'sp1', 'sp1']],
             'clusters': [[[4, 5], [9, 9], [11, 11]]],
             'constituents': [[1, 1, '0-I-C'], [2, 2, '1-love-C'],
                              [4, 4, '2-my-C'], [5, 5, '3-pet-C'],
                              [9, 9, '4-Anna-C'], [10, 10, '5-.-C'],
                              [11, 11, '6-She-C'], [12, 12, '7-is-C'],
                              [13, 13, '8-my-C'], [14, 14, '9-favorite-C'],
                              [15, 15, '10-.-C']],
             'ner': [[1, 1, '0-I-N'], [2, 2, '1-love-N'], [4, 4, '2-my-N'],
                     [5, 5, '3-pet-N'], [9, 9, '4-Anna-N'], [10, 10, '5-.-N'],
                     [11, 11, '6-She-N'], [12, 12, '7-is-N'], [13, 13, '8-my-N'],
                     [14, 14, '9-favorite-N'], [15, 15, '10-.-N']],
             'sample_id': None})
        self.assertEqual(
            a63.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['I', 'love', 'my', 'pet', 'Anna', '.'],
                           ['She', 'is', 'my', 'favorite', '.', ' haha', 'test']],
             'speakers': [['sp1', 'sp1', 'sp1', 'sp1', 'sp1', 'sp1'],
                          ['sp1', 'sp1', 'sp1', 'sp1', 'sp1', 'sp_ins', 'sp_ins']],
             'clusters': [[[2, 3], [4, 4], [6, 6]]],
             'constituents': [[0, 0, '0-I-C'], [1, 1, '1-love-C'],
                              [2, 2, '2-my-C'], [3, 3, '3-pet-C'],
                              [4, 4, '4-Anna-C'], [5, 5, '5-.-C'],
                              [6, 6, '6-She-C'], [7, 7, '7-is-C'],
                              [8, 8, '8-my-C'], [9, 9, '9-favorite-C'],
                              [10, 10, '10-.-C']],
             'ner': [[0, 0, '0-I-N'], [1, 1, '1-love-N'], [2, 2, '2-my-N'],
                     [3, 3, '3-pet-N'], [4, 4, '4-Anna-N'], [5, 5, '5-.-N'],
                     [6, 6, '6-She-N'], [7, 7, '7-is-N'], [8, 8, '8-my-N'],
                     [9, 9, '9-favorite-N'], [10, 10, '10-.-N']],
             'sample_id': None})
        self.assertEqual(
            a64.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['I', 'Wang', 'love', 'my', 'Xiao', 'pet', 'Anna',
                            '.'], ['She', 'is', 'fdu', 'jiangwan', 'cross_2',
                                   'my', 'favorite', '.']],
             'speakers': [['sp1', 'sp_ins', 'sp1', 'sp1', 'sp_ins', 'sp1',
                           'sp1', 'sp1'], ['sp1', 'sp1', 'sp_ins', 'sp_ins',
                                           'sp_ins', 'sp1', 'sp1', 'sp1']],
             'clusters': [[[3, 5], [6, 6], [8, 8]]],
             'constituents': [[0, 0, '0-I-C'], [2, 2, '1-love-C'],
                              [3, 3, '2-my-C'], [5, 5, '3-pet-C'],
                              [6, 6, '4-Anna-C'], [7, 7, '5-.-C'],
                              [8, 8, '6-She-C'], [9, 9, '7-is-C'],
                              [13, 13, '8-my-C'], [14, 14, '9-favorite-C'],
                              [15, 15, '10-.-C']],
             'ner': [[0, 0, '0-I-N'],  [2, 2, '1-love-N'], [3, 3, '2-my-N'],
                     [5, 5, '3-pet-N'], [6, 6, '4-Anna-N'], [7, 7, '5-.-N'],
                     [8, 8, '6-She-N'], [9, 9, '7-is-N'], [13, 13, '8-my-N'],
                     [14, 14, '9-favorite-N'], [15, 15, '10-.-N']],
             'sample_id': None})

    def test_delete(self):
        a1 = sample1
        a71 = a1.delete_field_at_indices('x', [0])
        a72 = a1.delete_field_at_indices('x', [0, [2, 4], len(a1.x)-1])
        self.assertEqual(
            a71.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['love', 'my', 'pet', 'Anna', '.'],
                           ['She', 'is', 'my', 'favorite', '.']],
             'speakers': [['sp1', 'sp1', 'sp1', 'sp1', 'sp1'],
                          ['sp1', 'sp1', 'sp1', 'sp1', 'sp1']],
             'clusters': [[[1, 2], [3, 3], [5, 5]]],
             'constituents': [[0, 0, '1-love-C'], [1, 1, '2-my-C'],
                              [2, 2, '3-pet-C'], [3, 3, '4-Anna-C'],
                              [4, 4, '5-.-C'], [5, 5, '6-She-C'],
                              [6, 6, '7-is-C'], [7, 7, '8-my-C'],
                              [8, 8, '9-favorite-C'], [9, 9, '10-.-C']],
             'ner': [[0, 0, '1-love-N'], [1, 1, '2-my-N'], [2, 2, '3-pet-N'],
                     [3, 3, '4-Anna-N'], [4, 4, '5-.-N'], [5, 5, '6-She-N'],
                     [6, 6, '7-is-N'], [7, 7, '8-my-N'], [8, 8, '9-favorite-N'],
                     [9, 9, '10-.-N']], 'sample_id': None})
        self.assertEqual(
            a72.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['love', 'Anna', '.'],
                           ['She', 'is', 'my', 'favorite']],
             'speakers': [['sp1', 'sp1', 'sp1'], ['sp1', 'sp1', 'sp1', 'sp1']],
             'clusters': [[[1, 1], [3, 3]]],
             'constituents': [[0, 0, '1-love-C'],
                              [1, 1, '4-Anna-C'], [2, 2, '5-.-C'],
                              [3, 3, '6-She-C'], [4, 4, '7-is-C'],
                              [5, 5, '8-my-C'], [6, 6, '9-favorite-C']],
             'ner': [[0, 0, '1-love-N'], [1, 1, '4-Anna-N'], [2, 2, '5-.-N'],
                     [3, 3, '6-She-N'], [4, 4, '7-is-N'], [5, 5, '8-my-N'],
                     [6, 6, '9-favorite-N']], 'sample_id': None})

    def test_insert(self):
        a1 = sample1
        a81 = a1.replace_field_at_indices('x', [0], ['$'])
        a82 = a1.replace_field_at_indices(
            'x', [0, [2, 4], [7, 8]], ['$', ['wang', 'xiao'], 'fDu'])
        self.assertEqual(
            a81.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['$', 'love', 'my', 'pet', 'Anna', '.'],
                           ['She', 'is', 'my', 'favorite', '.']],
             'speakers': [['sp_repl', 'sp1', 'sp1', 'sp1', 'sp1', 'sp1'],
                          ['sp1', 'sp1', 'sp1', 'sp1', 'sp1']],
             'clusters': [[[2, 3], [4, 4], [6, 6]]],
             'constituents': [[0, 0, '0-I-C'], [1, 1, '1-love-C'],
                              [2, 2, '2-my-C'], [3, 3, '3-pet-C'],
                              [4, 4, '4-Anna-C'], [5, 5, '5-.-C'],
                              [6, 6, '6-She-C'], [7, 7, '7-is-C'],
                              [8, 8, '8-my-C'], [9, 9, '9-favorite-C'],
                              [10, 10, '10-.-C']],
             'ner': [[0, 0, '0-I-N'], [1, 1, '1-love-N'], [2, 2, '2-my-N'],
                     [3, 3, '3-pet-N'], [4, 4, '4-Anna-N'], [5, 5, '5-.-N'],
                     [6, 6, '6-She-N'], [7, 7, '7-is-N'], [8, 8, '8-my-N'],
                     [9, 9, '9-favorite-N'], [10, 10, '10-.-N']],
             'sample_id': None})
        self.assertEqual(
            a82.dump(),
            {'doc_key': 'doc_key1',
             'sentences': [['$', 'love', 'wang', 'xiao', 'Anna', '.'],
                           ['She', 'fDu', 'my', 'favorite', '.']],
             'speakers': [['sp_repl', 'sp1', 'sp_repl', 'sp_repl', 'sp1', 'sp1'],
                          ['sp1', 'sp_repl', 'sp1', 'sp1', 'sp1']],
             'clusters': [[[2, 3], [4, 4], [6, 6]]],
             'constituents': [[0, 0, '0-I-C'], [1, 1, '1-love-C'],
                              [2, 2, '2-my-C'], [3, 3, '3-pet-C'],
                              [4, 4, '4-Anna-C'], [5, 5, '5-.-C'],
                              [6, 6, '6-She-C'], [7, 7, '7-is-C'],
                              [8, 8, '8-my-C'], [9, 9, '9-favorite-C'],
                              [10, 10, '10-.-C']],
             'ner': [[0, 0, '0-I-N'], [1, 1, '1-love-N'], [2, 2, '2-my-N'],
                     [3, 3, '3-pet-N'], [4, 4, '4-Anna-N'], [5, 5, '5-.-N'],
                     [6, 6, '6-She-N'], [7, 7, '7-is-N'], [8, 8, '8-my-N'],
                     [9, 9, '9-favorite-N'], [10, 10, '10-.-N']],
             'sample_id': None})

    def test_gets(self):
        for sample in samples:
            self.assertEqual(sample.get_value('x'), sample.x.words)
            self.assertEqual(sample.get_value('clusters'),
                             sample.clusters.field_value)
            self.assertEqual(sample.get_words('x'), sample.x.words)
            self.assertEqual(len(sample.get_words('x')),
                             len(sample.get_mask('x')))


if __name__ == "__main__":
    unittest.main()
