import unittest

from textflint.input.component.sample.ner_sample import *

data = {'x': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British',
              'lamb', '.'],
        'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC', 'O', 'O']}
ner_sample = NERSample(data)


class TestNERSample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, NERSample, {'x': 'apple'})
        self.assertRaises(AssertionError, NERSample, {'y': 'apple'})
        self.assertRaises(AssertionError, NERSample, {'x': ''})
        self.assertRaises(AssertionError, NERSample,
                          {'x': ['EU', 'rejects', 'German', 'call', 'to',
                                 'boycott', 'British', 'lamb', '.'],
                           'y': ['E-ORG', 'O', 'B-MISC', 'O', 'O', 'O',
                                 'B-MISC', 'O', 'O']})
        self.assertRaises(AssertionError, NERSample, {'x': 'the US', 'y':
            ['I-LOC', 'B-LOC']})

    def test_insert_field_before_index(self):
        # test insert before index and mask
        ins_bef = ner_sample.insert_field_before_index('text', 0, '$$$')
        self.assertEqual(['$$$', 'EU', 'rejects', 'German', 'call', 'to',
                          'boycott', 'British', 'lamb', '.'],
                         ins_bef.dump()['x'])
        self.assertEqual(ins_bef.dump()['y'], ['O'] + ner_sample.dump()['y'])
        self.assertEqual([2, 0, 0, 0, 0, 0, 0, 0, 0, 0], ins_bef.text.mask)

    def test_insert_field_after_index(self):
        # test insert before index and mask
        ins_aft = ner_sample.insert_field_after_index('text', 2, '$$$')
        self.assertEqual(['EU', 'rejects', 'German', '$$$', 'call', 'to',
                          'boycott', 'British', 'lamb', '.'],
                         ins_aft.dump()['x'])
        self.assertEqual(
            ins_aft.dump()['y'],
            ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O'])
        self.assertEqual([0, 0, 0, 2, 0, 0, 0, 0, 0, 0], ins_aft.text.mask)

    def test_delete_field_at_index(self):
        # test insert before index and mask
        del_sample = ner_sample.delete_field_at_index('text', [1, 3])
        self.assertEqual(['EU', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
                         del_sample.dump()['x'])
        self.assertEqual(del_sample.dump()['y'],
                         ['B-ORG', 'O', 'O', 'O', 'B-MISC', 'O', 'O'])
        self.assertEqual([0, 0, 0, 0, 0, 0, 0], del_sample.text.mask)

    def test_find_entities_BIO(self):
        # test find_entities_BIO
        entities = ner_sample.find_entities_BIO(ner_sample.dump()['x'],
                                                ner_sample.dump()['y'])
        self.assertEqual([{'start': 0, 'end': 0,
                           'entity': 'EU', 'tag': 'ORG'},
                          {'start': 2, 'end': 2,
                           'entity': 'German', 'tag': 'MISC'},
                          {'start': 6, 'end': 6,
                           'entity': 'British', 'tag': 'MISC'}], entities)

    def test_entity_replace(self):
        # test wrong input
        self.assertRaises(AssertionError, ner_sample.entity_replace,
                          2, 1, 'Beijing', 'Loc')

        # test entity_replace
        new = ner_sample.entity_replace(0, 0, 'New York', 'Loc')
        self.assertEqual(new.dump()['x'],
                         ['New', 'York', 'rejects', 'German', 'call', 'to',
                          'boycott', 'British', 'lamb', '.'])
        self.assertEqual(new.dump()['y'], ['B-Loc', 'I-Loc', 'O', 'B-MISC',
                                           'O', 'O', 'O', 'B-MISC', 'O', 'O'])
        # TODO wait repair bug
        print(new.text.mask)

    def test_entities_replace(self):
        # test wrong data
        self.assertRaises(AssertionError, ner_sample.entities_replace,
                          [], ['aaa'])
        self.assertRaises(AssertionError, ner_sample.entities_replace,
                          [{'start': 2, 'end': 2, 'entity': 'bbb',
                            'tag': 'MISC'}], [])
        self.assertRaises(AssertionError, ner_sample.entities_replace,
                          {'start': 2, 'end': 2, 'entity': 'bbb',
                           'tag': 'MISC'}, ['aaa'])
        self.assertRaises(AssertionError, ner_sample.entities_replace,
                          [{'start': 2, 'end': 2, 'entity': 'bbb',
                            'tag': 'MISC'}], 'aaa')

        # test entities_replace
        new = ner_sample.entities_replace([{'start': 0, 'end': 0,
                                            'entity': 'aaa', 'tag': 'PRE'},
                                           {'start': 2, 'end': 2,
                                            'entity': 'bbb', 'tag': 'MISC'}],
                                          ['aaa', 'bbb'])
        self.assertEqual(['aaa', 'rejects', 'bbb', 'call', 'to', 'boycott',
                          'British', 'lamb', '.'], new.dump()['x'])
        self.assertEqual(['B-PRE', 'O', 'B-MISC', 'O', 'O', 'O', 'B-MISC',
                          'O', 'O'], new.dump()['y'])
        # TODO wait repair bug
        print(new.text.mask)

    def test_mask(self):
        # test does the mask label work
        new = ner_sample.insert_field_after_index('text', 0, 'abc')
        new = new.delete_field_at_index('text', 1)
        # TODO wait repair bug
        print(new.dump())
        print(new.text.mask)

    def test_get_words(self):
        # test get words
        self.assertEqual(['EU', 'rejects', 'German', 'call',
                          'to', 'boycott', 'British',
                          'lamb', '.'], ner_sample.get_words('text'))
        self.assertRaises(AssertionError, ner_sample.get_words, 'tags')

    def test_get_text(self):
        # test get text
        self.assertEqual('EU rejects German call to boycott '
                         'British lamb.', ner_sample.get_text('text'))
        self.assertRaises(AssertionError, ner_sample.get_text, 'tags')

    def test_get_value(self):
        # test get value
        self.assertEqual(['EU', 'rejects', 'German', 'call', 'to',
                          'boycott', 'British', 'lamb', '.'],
                         ner_sample.get_value('text'))
        self.assertEqual(['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O',
                          'B-MISC', 'O', 'O'], ner_sample.get_value('tags'))

    def test_dump(self):
        # test dump
        self.assertEqual({'x': ['EU', 'rejects', 'German', 'call', 'to',
                                'boycott', 'British', 'lamb', '.'],
                          'y': ['B-ORG', 'O', 'B-MISC', 'O', 'O', 'O',
                                'B-MISC', 'O', 'O'], 'sample_id': None},
                         ner_sample.dump())


if __name__ == "__main__":
    unittest.main()
