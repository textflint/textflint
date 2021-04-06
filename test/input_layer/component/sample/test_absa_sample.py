import unittest

from textflint.input_layer.component.sample.absa_sample import *

data = {"sentence": "The bread is top notch as well.",
        "term_list": {
            "32897564#894393#2_0":
                {"id": "32897564#894393#2_0",
                 "polarity": "positive",
                 "term": "bread",
                 "from": 4,
                 "to": 9,
                 "opinion_words": ["top notch"],
                 "opinion_position": [[13, 22]]}
        }
        }


class TestABSASample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(
            AssertionError, ABSASample,
            {'x': 'food', "term_list": {"32897564#894393#2_0":
                                            {"polarity": "positive",
                                             "term": "bread", "from": 4,
                                             "to": 9, "opinion_words":
                                                 ["top notch"],
                                             "opinion_position": [[13, 22]]}}})
        self.assertRaises(
            ValueError, ABSASample,
            {'sentence': '', "term_list":
                {"32897564#894393#2_0":
                     {"polarity": "positive", "term": "bread", "from": 4,
                      "to": 9, "opinion_words": ["top notch"],
                      "opinion_position": [[13, 22]]}}})
        self.assertRaises(
            AssertionError, ABSASample,
            {'sentence': ['food'], "term_list":
                {"32897564#894393#2_0":
                     {"polarity": "positive", "term": "bread", "from": 4,
                      "to": 9, "opinion_words": ["top notch"],
                      "opinion_position": [[13, 22]]}}})
        self.assertRaises(
            AssertionError, ABSASample, {'sentence': 'food', "term_list": ''})
        self.assertRaises(
            AssertionError, ABSASample,
            {'sentence': 'food', "term_list":
                {"32897564#894393#2_0": [{"polarity": "positive",
                                          "term": "bread", "from": 4, "to": 9,
                                          "opinion_words": ["top notch"],
                                          "opinion_position": [[13, 22]]}]}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "",
                                                       "term": "bread",
                                                       "from": 4, "to": 9,
                                                       "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position":
                                                           [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0":
                                   {"polarity": 5, "term": "bread",
                                    "from": 4, "to": 9, "opinion_words":
                                        ["top notch"], "opinion_position":
                                        [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0":
                                   {"polarity": ["positive"], "term": "bread",
                                    "from": 4, "to": 9, "opinion_words":
                                        ["top notch"], "opinion_position":
                                        [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "", "from": 4,
                                                       "to": 9, "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position":
                                                           [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": 5, "from": 4,
                                                       "to": 9, "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position":
                                                           [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": ["bread"],
                                                       "from": 4, "to": 9,
                                                       "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position":
                                                           [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "bread", "from":
                                                           "", "to": 9,
                                                       "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position":
                                                           [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "bread",
                                                       "from": [], "to": 9,
                                                       "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position":
                                                           [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "bread",
                                                       "from": -5, "to": 9,
                                                       "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position":
                                                           [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0":
                                   {"polarity": "positive", "term": "bread",
                                    "from": 4, "to": "", "opinion_words":
                                        ["top notch"], "opinion_position":
                                        [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0":
                                   {"polarity": "positive", "term": "bread",
                                    "from": 4, "to": [], "opinion_words":
                                        ["top notch"], "opinion_position":
                                        [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0":
                                   {"polarity": "positive", "term": "bread",
                                    "from": 4, "to": -5, "opinion_words":
                                        ["top notch"], "opinion_position":
                                        [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "bread",
                                                       "from": 4, "to": 9,
                                                       "opinion_words": [],
                                                       "opinion_position":
                                                           [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "bread",
                                                       "from": 4, "to": 9,
                                                       "opinion_words": [''],
                                                       "opinion_position":
                                                           [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "bread",
                                                       "from": 4, "to": 9,
                                                       "opinion_words": [-5],
                                                       "opinion_position":
                                                           [[13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "bread",
                                                       "from": 4, "to": 9,
                                                       "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position":
                                                           [[13, 11]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "bread",
                                                       "from": 4, "to": 9,
                                                       "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position":
                                                           [[-13, 22]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "bread",
                                                       "from": 4, "to": 9,
                                                       "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position": []}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list":
                              {"32897564#894393#2_0": {"polarity": "positive",
                                                       "term": "bread",
                                                       "from": 4, "to": 9,
                                                       "opinion_words":
                                                           ["top notch"],
                                                       "opinion_position":
                                                           [[]]}}})
        self.assertRaises(AssertionError, ABSASample, {
            'sentence': 'food', "term_list": {"32897564#894393#2_0":
                                                  {"polarity": "positive",
                                                   "term": "bread", "from": 4,
                                                   "to": 9, "opinion_words":
                                                       ["top notch"],
                                                   "opinion_position":
                                                       [[13, 22, 33]]}}})
        self.assertRaises(AssertionError, ABSASample,
                          {'sentence': 'food', "term_list": {
                              "32897564#894393#2_0": {"polarity": "positive",
                                                      "term": "bread", "from": 4,
                                                      "to": 9, "opinion_words":
                                                          ["top notch"],
                                                      "opinion_position": ['']}}})

    def test_insert_field_before_index(self):
        absa_sample = ABSASample(data)
        # test insert before index and mask
        ins_aft = absa_sample.insert_field_before_index('sentence', 2, '$$$')
        self.assertEqual('The bread $$$ is top notch as well.',
                         ins_aft.get_text('sentence'))
        self.assertEqual(
            [0, 0, 2, 0, 0, 0, 0, 0, 0], ins_aft.get_mask('sentence'))

    def test_insert_field_after_index(self):
        absa_sample = ABSASample(data)
        # test insert before index and mask
        ins_aft = absa_sample.insert_field_after_index('sentence', 2, '$$$')
        self.assertEqual('The bread is $$$ top notch as well.',
                         ins_aft.get_text('sentence'))
        self.assertEqual(
            [0, 0, 0, 2, 0, 0, 0, 0, 0], ins_aft.get_mask('sentence'))

    def test_delete_field_at_index(self):
        absa_sample = ABSASample(data)
        # test insert before index and mask
        del_sample = absa_sample.delete_field_at_index('sentence', [1, 3])
        self.assertEqual(
            ['The', 'top', 'notch', 'as', 'well', '.'],
            del_sample.get_value('sentence'))
        self.assertEqual([0, 0, 0, 0, 0, 0], del_sample.get_mask('sentence'))   
    
    def test_unequal_replace_field_at_indices(self):
        absa_sample = ABSASample(data)
        # test insert before index and mask
        ins_aft = absa_sample.unequal_replace_field_at_indices(
            'sentence', [2], [['$$$', '123']])
        self.assertEqual('The bread $$$ 123 top notch as well.',
                         ins_aft.get_text('sentence'))
        self.assertEqual([0, 0, 2, 2, 0, 0, 0, 0, 0],
                         ins_aft.get_mask('sentence'))

    def test_mask(self):
        absa_sample = ABSASample(data)
        # test does the mask label work
        new = absa_sample.insert_field_after_index('sentence', 0, 'abc')
        new = new.delete_field_at_index('sentence', 1)
        # TODO wait repair bug
        print(new.dump())
        print(new.get_mask('sentence'))

    def test_get_words(self):
        absa_sample = ABSASample(data)
        # test get words
        print(absa_sample.get_words('sentence'))
        self.assertEqual(['The', 'bread', 'is', 'top', 'notch', 'as',
                          'well', '.'], absa_sample.get_words('sentence'))

    def test_get_text(self):
        absa_sample = ABSASample(data)
        # test get text
        self.assertEqual('The bread is top notch as well.',
                         absa_sample.get_text('sentence'))

    def test_get_value(self):
        absa_sample = ABSASample(data)
        # test get value
        self.assertEqual('The bread is top notch as well.',
                         absa_sample.get_text('sentence'))

    def test_dump(self):
        # test dump
        absa_sample = ABSASample(data)
        print(absa_sample.dump())
        self.assertEqual({'sentence': 'The bread is top notch as well.',
                          'term_list': {'32897564#894393#2_0':
                                            {'id': '32897564#894393#2_0',
                                             'polarity': 'positive',
                                             'term': 'bread', 'from': 4, 'to': 9,
                                             'opinion_words': ['top notch'],
                                             'opinion_position': [[13, 22]]}},
                          'contra': None, 'multi': None, 'id': None,
                          'trans_id': None, 'sample_id': None},
                         absa_sample.dump())


if __name__ == "__main__":
    unittest.main()
