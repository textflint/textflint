import unittest
from textflint.input_layer.dataset import Dataset
from textflint.generation_layer.generator.mrc_generator import MRCGenerator
from textflint.input_layer.component.sample import MRCSample

context = 'Super Bowl 50 was an American football game to determine ' \
          'the champion of the National Football League ' \
          '(NFL) for the 2015 season. The American Football ' \
          'Conference (AFC) champion ' \
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
          'so that the logo could prominently feature the Arabic numerals ' \
          '50.! @ # $ % ^ & * ( )'
data_sample = MRCSample(
    {'context': context, 'question': 'Which NFL team represented the '
                                     'AFC at Super Bowl 50?',
        'answers': [{"text": "Denver Broncos", "answer_start": 177},
                    {"text": "Denver Broncos", "answer_start": 177},
                    {"text": "Denver Broncos", "answer_start": 177}],
        'title': "Super_Bowl_50", 'is_impossible': False})
sample2 = MRCSample(
    {'context': " ", 'question': 'Which NFL team represented '
                                 'the AFC at Super Bowl 50?',
        'answers': [], 'title': "Super_Bowl_50", 'is_impossible': True})
sample3 = MRCSample(
    {'context': "! @ # $ % ^ & * ( )",
     'question': 'Which NFL team represented the AFC at Super Bowl 50?',
        'answers': [], 'title': "Super_Bowl_50", 'is_impossible': True})

dataset = Dataset('MRC')
dataset.load(data_sample)
dataset.extend([sample2, sample3])


class TestMRCGenerator(unittest.TestCase):

    def test_generate(self):
        # test task transformation
        # TODO, domain transformation addsentdiverse
        trans_methods = ["PerturbAnswer", "ModifyPos"]
        gene = MRCGenerator(trans_methods=trans_methods,
                            sub_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(1, len(trans_rst))
            for index in range(len(original_samples)):
                ori_sample = original_samples[index]
                tran_sample = trans_rst[index]
                self.assertEqual(ori_sample.get_words('question'),
                                 tran_sample.get_words('question'))
                ori_answers = ori_sample.get_answers()
                tran_answers = tran_sample.dump()['answers']
                for i in range(len(ori_answers)):
                    self.assertEqual(ori_answers[i]['text'],
                                     tran_answers[i]['text'])

        # test wrong trans_methods
        gene = MRCGenerator(trans_methods=["wrong_transform_method"],
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = MRCGenerator(trans_methods=["AddSubtree"],
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = MRCGenerator(trans_methods="OOV",
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))


if __name__ == "__main__":
    unittest.main()
