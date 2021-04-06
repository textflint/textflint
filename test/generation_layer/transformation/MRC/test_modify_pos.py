import unittest

from textflint.input_layer.component.sample.mrc_sample import MRCSample
from textflint.generation_layer.transformation.MRC.modify_pos \
    import ModifyPos

context = 'Super Bowl 50 was an American football game to determine the ' \
          'champion of the National Football League ' \
          '(NFL) for the 2015 season. The American Football ' \
          'Conference (AFC) champion ' \
          'Denver Broncos defeated the National Football Conference ' \
          '(NFC) champion Carolina Panthers 24–10 ' \
          'to earn their third Super Bowl title. The game was ' \
          'played on February 7, 2016, at Levi\'s Stadium ' \
          'in the San Francisco Bay Area at Santa Clara, California. ' \
          'As this was the 50th Super Bowl, ' \
          'the league emphasized the "golden anniversary" with various ' \
          'gold-themed initiatives, ' \
          'as well as temporarily suspending the tradition of naming each ' \
          'Super Bowl game with Roman numerals ' \
          '(under which the game would have been known as "Super Bowl L"), ' \
          'so that the logo could prominently feature the Arabic numerals 50.'
data_sample = MRCSample(
        {'context': context, 'question': 'Which NFL team '
                                         'represented the AFC at Super Bowl 50?',
         'answers': [{"text": "Denver Broncos", "answer_start": 177},
                     {"text": "Denver Broncos", "answer_start": 177},
                     {"text": "Denver Broncos", "answer_start": 177}],
         'title': "Super_Bowl_50", 'is_impossible': False})
transformation = ModifyPos()


class TestModifyPosition(unittest.TestCase):

    def test_ModifyPosition(self):
        
        change = transformation.transform(data_sample, n=1)
        self.assertEqual(len(change), 1)
        original_sents = data_sample.get_sentences('context')
        trans_sents = change[0].get_sentences('context')
        changed = False
        for i in range(len(original_sents)):
            if trans_sents[i] != original_sents[i]:
                changed = True
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
