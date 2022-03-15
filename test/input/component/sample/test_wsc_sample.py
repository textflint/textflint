import unittest

from textflint.input.component.sample.wsc_sample import *

data = {"text": "The city councilmen refused the demonstrators a permit because"
                " they feared violence.",
        'target': {"noun1": "The city councilmen", "noun2": "The demonstrators",
                   "noun1_idx": 0, "noun2_idx": 4, "pron": "they", "pron_idx": 9},
        "label": 0, "index": 0}
wsc_sample = WSCSample(data)


class TestWSCSample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, WSCSample, {'text': 'the city'})
        self.assertRaises(AssertionError, WSCSample, {'target': 'noun'})
        self.assertRaises(AssertionError, WSCSample, {'text': ''})
        self.assertRaises(AssertionError, WSCSample, {"text": "The city councilmen refused "
                                                              "the demonstrators a permit because"
                                                              " they feared violence.",
                                                      "target": {"noun1": "The city councilmen",
                                                                 "noun2": "The demonstrators"}
                           })
        self.assertRaises(AssertionError, WSCSample, {"text": "The city councilmen refused "
                                                              "the demonstrators a permit because"
                                                              " they feared violence.",
                                                  "target": {"noun1": "The city councilmen",
                                                                 "noun2": "The demonstrators",
                                                                 "noun1_idx": 0, "noun2_idx": 4,
                                                                 "pron": "they", "pron_idx": 9},
                                                  "label": 'A', "index": 0})
        self.assertRaises(ValueError, WSCSample, {"text": "The city councilmen refused "
                                                              "the demonstrators a permit because"
                                                              " they feared violence.",
                                                  "target": {"noun1": "The city councilmen",
                                                                 "noun2": "The demonstrators",
                                                                 "noun1_idx": -1, "noun2_idx": 80,
                                                                 "pron": "they", "pron_idx": 9},
                                                  "label": 0, "index": 0})

    def test_dump(self):
        # test dump
        self.assertEqual(data, wsc_sample.dump())

    def test_insert_field_before_index(self):
        # test insert before index and mask
        ins_bef = wsc_sample.insert_field_before_index('text', 0, '[cls]')
        self.assertEqual("[cls] The city councilmen refused the demonstrators a permit because "
                         "they feared violence.",
                         ins_bef.dump()['text'])
        self.assertEqual(ins_bef.dump()['target'],
                         {"noun1": "The city councilmen", "noun2": "The demonstrators",
                          "noun1_idx": 1, "noun2_idx": 5, "pron": "they", "pron_idx": 10}
                         )

    def test_insert_field_after_index(self):
        # test insert after index and mask
        ins_aft = wsc_sample.insert_field_after_index('text', 2, '$$$')
        self.assertEqual("The city councilmen $$$ refused the demonstrators a permit because "
                         "they feared violence.",
                         ins_aft.dump()['text'])
        self.assertEqual(ins_aft.dump()['target'],
                         {"noun1": "The city councilmen", "noun2": "The demonstrators",
                          "noun1_idx": 0, "noun2_idx": 5, "pron": "they", "pron_idx": 10}
                         )

    def test_delete_field_at_index(self):
        # test delete at index
        del_sample = wsc_sample.delete_field_at_index('text', 6)
        self.assertTrue(del_sample.is_legal())
        self.assertEqual("The city councilmen refused the demonstrators permit because "
                         "they feared violence.",
                         del_sample.dump()['text'])
        self.assertEqual(del_sample.dump()['target'],
                         {"noun1": "The city councilmen", "noun2": "The demonstrators",
                          "noun1_idx": 0, "noun2_idx": 4, "pron": "they", "pron_idx": 8}
                         )

    def test_get_words(self):
        # test get words
        self.assertEqual(['The', 'city', 'councilmen', 'refused', 'the', 'demonstrators', 'a',
                          'permit', 'because', 'they', 'feared', 'violence', '.'],
                         wsc_sample.get_words('text'))

    def test_get_text(self):
        # test get text
        self.assertEqual(
            data['text'],
            wsc_sample.get_text('text'))

    def test_get_value(self):
        # test get value
        self.assertEqual(
            data['text'],
            wsc_sample.get_value('text'))


if __name__ == "__main__":
    unittest.main()
