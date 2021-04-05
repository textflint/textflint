import unittest

from TextFlint.input_layer.component.sample.mrc_sample import MRCSample

context = 'Super Bowl 50 was an American football game to determine the champion ' \
          'of the National Football League ' \
          '(NFL) for the 2015 season. The American Football Conference (AFC)' \
          ' champion ' \
          'Denver Broncos defeated the National Football Conference (NFC) ' \
          'champion Carolina Panthers 24–10 ' \
          'to earn their third Super Bowl title. The game was played on ' \
          'February 7, 2016, at Levi\'s Stadium ' \
          'in the San Francisco Bay Area at Santa Clara, California. As ' \
          'this was the 50th Super Bowl, ' \
          'the league emphasized the "golden anniversary" with various ' \
          'gold-themed initiatives, ' \
          'as well as temporarily suspending the tradition of naming each ' \
          'Super Bowl game with Roman numerals ' \
          '(under which the game would have been known as "Super Bowl L"), ' \
          'so that the logo could prominently feature the Arabic numerals 50.'
question = 'Which NFL team represented the AFC at Super Bowl 50?'
answers = [{"text": "Denver Broncos",
            "answer_start": 177},
           {"text": "Denver Broncos",
            "answer_start": 177},
           {"text": "Denver Broncos",
            "answer_start": 177}]
title = "Super_Bowl_50"
is_impossible = False
mrc_sample = MRCSample({'context': context,
                        'question': 'Which NFL team represented the AFC '
                                    'at Super Bowl 50?',
                        'answers': [{"text": "Denver Broncos",
                                     "answer_start": 177},
                                    {"text": "Denver Broncos",
                                     "answer_start": 177},
                                    {"text": "Denver Broncos",
                                     "answer_start": 177}],
                        'title': "Super_Bowl_50",
                        'is_impossible': False})


class TestMRCSample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, MRCSample, {'context': context})
        self.assertRaises(AssertionError, MRCSample, {'question': question})
        self.assertRaises(AssertionError, MRCSample, {'context': ''})
        self.assertRaises(AssertionError,
                          MRCSample,
                          {'context': context,
                           'question': question,
                           'answers': answers[0],
                           'title': title,
                           'is_impossible': is_impossible})
        self.assertRaises(ValueError,
                          MRCSample,
                          {'context': context,
                           'question': question,
                           'answers': [{'text': "Denver Broncos",
                                        'answer_start': 170}],
                           'title': title,
                           'is_impossible': is_impossible})
        self.assertRaises(ValueError,
                          MRCSample,
                          {'context': context,
                           'question': question,
                           'answers': answers,
                           'title': [],
                           'is_impossible': is_impossible})

    def test_convert_idx(self):
        spans = mrc_sample.convert_idx(context, mrc_sample.get_words('context'))
        self.assertEqual(spans[1][0], 6)

    def test_insert_field_after_index(self):
        # test insert after index and mask
        ins_aft = mrc_sample.insert_field_after_index(
            'context', 2, 'TextFlint')
        self.assertTrue(ins_aft.is_legal())
        self.assertEqual(
            'Super Bowl 50 TextFlint was an American football game to '
            'determine the champion '
            'of the National Football League (NFL) for the 2015 season.',
            ins_aft.get_sentences('context')[0])

    def test_insert_field_before_index(self):
        # test insert before index and mask
        ins_bef = mrc_sample.insert_field_before_index(
            'context', 2, 'TextFlint')
        self.assertTrue(ins_bef.is_legal())
        self.assertEqual(
            'Super Bowl TextFlint 50 was an American football game to '
            'determine the champion '
            'of the National Football League (NFL) for the 2015 season.',
            ins_bef.get_sentences('context')[0])

    def test_delete_field_at_index(self):
        # test delete at index
        delete = mrc_sample.delete_field_at_index('context', 2)
        self.assertTrue(delete.is_legal())
        self.assertEqual(
            'Super Bowl was an American football game to '
            'determine the champion '
            'of the National Football League (NFL) for the 2015 season.',
            delete.get_sentences('context')[0])

    def test_unequal_replace_field_at_indices(self):
        # test unequal replacement
        replace = mrc_sample.unequal_replace_field_at_indices(
            'context', [2], [['TextFlint', 'software']])
        self.assertTrue(replace.is_legal())
        self.assertEqual(
            'Super Bowl TextFlint software was an American football '
            'game to determine the champion '
            'of the National Football League (NFL) for the 2015 season.',
            replace.get_sentences('context')[0])

    def test_get_words(self):
        # test get words
        self.assertEqual(['Which',
                          'NFL',
                          'team',
                          'represented',
                          'the',
                          'AFC',
                          'at',
                          'Super',
                          'Bowl',
                          '50',
                          '?'],
                         mrc_sample.get_words('question'))

    def test_get_text(self):
        # test get text
        self.assertEqual(
            'Which NFL team represented the AFC at Super Bowl 50?',
            mrc_sample.get_text('question'))

    def test_get_value(self):
        # test get value
        self.assertEqual(
            'Which NFL team represented the AFC at Super Bowl 50?',
            mrc_sample.get_value('question'))

    def test_dump(self):
        # test dump
        self.assertEqual({'context': 'Super Bowl 50 was an American football '
                                     'game to determine the champion '
                                     'of the National Football League (NFL) '
                                     'for the 2015 season. '
                                     'The American Football Conference (AFC) '
                                     'champion Denver Broncos '
                                     'defeated the National Football Conference'
                                     ' (NFC) champion '
                                     'Carolina Panthers 24–10 to earn their '
                                     'third Super Bowl title. '
                                     'The game was played on February 7, 2016, '
                                     'at Levi\'s Stadium '
                                     'in the San Francisco Bay Area at Santa '
                                     'Clara, California. '
                                     'As this was the 50th Super Bowl, the '
                                     'league emphasized '
                                     'the "golden anniversary" with various '
                                     'gold-themed initiatives, '
                                     'as well as temporarily suspending the '
                                     'tradition of '
                                     'naming each Super Bowl game with Roman '
                                     'numerals '
                                     '(under which the game would have been '
                                     'known as "Super Bowl L"), '
                                     'so that the logo could prominently '
                                     'feature the Arabic numerals 50.',
                          'question': 'Which NFL team represented the AFC'
                                      ' at Super Bowl 50?',
                          'answers': [{'text': 'Denver Broncos',
                                       'answer_start': 177},
                                      {'text': 'Denver Broncos',
                                       'answer_start': 177},
                                      {'text': 'Denver Broncos',
                                       'answer_start': 177}],
                          'title': 'Super_Bowl_50', 'is_impossible': False,
                          'sample_id': None},
                         mrc_sample.dump())


if __name__ == "__main__":
    unittest.main()
