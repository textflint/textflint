import unittest
from textflint.input_layer.model.metrics.metrics import *


class TestMetrics(unittest.TestCase):
    def test_squad_metric(self):
        predictions = [
            {'prediction_text': '1976', 'id': '56e10a3be3433e1400422b22'}]
        references = [{'answers': {'answer_start': [97], 'text': ['1976']},
                       'id': '56e10a3be3433e1400422b22'}]
        squad_metric = SQuADMetric()
        score = squad_metric(predictions, references)
        self.assertEqual(100.0, score['exact_match'])
        self.assertEqual(100.0, score['f1'])


if __name__ == '__main__':
    unittest.main()
