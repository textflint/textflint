import unittest

from textflint.input_layer.component.sample.mrc_sample import MRCSample
from textflint.generation_layer.transformation.MRC import PerturbAnswer
from textflint.common.settings import NEAR_DICT_PATH, POS_DICT_PATH
from textflint.common.utils.install import download_if_needed
from textflint.common.utils import load_supporting_file

seed = 123

nearby_word_dict, pos_tag_dict = load_supporting_file(
    download_if_needed(NEAR_DICT_PATH),
    download_if_needed(POS_DICT_PATH))

context = 'Super Bowl 50 was an American football game to determine ' \
          'the champion of the National Football League ' \
          '(NFL) for the 2015 season. The American Football ' \
          'Conference (AFC) champion ' \
          'Denver Broncos defeated the National Football Conference ' \
          '(NFC) champion Carolina Panthers 24–10 ' \
          'to earn their third Super Bowl title. The game was played on ' \
          'February 7, 2016, at Levi\'s Stadium ' \
          'in the San Francisco Bay Area at Santa Clara, California. ' \
          'As this was the 50th Super Bowl, ' \
          'the league emphasized the "golden anniversary" with various ' \
          'gold-themed initiatives, ' \
          'as well as temporarily suspending the tradition of namin' \
          'g each Super Bowl game with Roman numerals ' \
          '(under which the game would have been known as "Super Bowl L"), ' \
          'so that the logo could prominently feature the Arabic numerals 50.'
data_sample = MRCSample(
    {'context': context, 'question': 'Which NFL team repre'
                                     'sented the AFC at Super Bowl 50?',
     'answers': [{"text": "Denver Broncos", "answer_start": 177},
                 {"text": "Denver Broncos", "answer_start": 177},
                 {"text": "Denver Broncos", "answer_start": 177}],
     'title': "Super_Bowl_50", 'is_impossible': False})
transformation = PerturbAnswer()


class TestPerturbAnswer(unittest.TestCase):

    def test_PerturbAnswer(self):

        change = transformation.transform(data_sample, n=1,
                                          nearby_word_dict=nearby_word_dict,
                                          pos_tag_dict=pos_tag_dict)
        sent_start = 0
        trans_sents = change[0].get_sentences('context')
        answer_token_start = change[0].get_answers()[0]['start']
        changed_sent = 'The British Soccer Meeting (AFC) champ Denver' \
                       ' Broncos defeated the State Soccer Meeting ' \
                       '(NFC) champ Michigan Vikings ' \
                       '24–10 to earn their 3rd Champ Bowl rubric.'

        for i, sent in enumerate(trans_sents):
            sent_len = len(PerturbAnswer.processor.tokenize(sent))
            if sent_start + sent_len <= answer_token_start:
                sent_start += sent_len
                continue
            self.assertTrue(changed_sent == sent)
            break


if __name__ == "__main__":
    unittest.main()
