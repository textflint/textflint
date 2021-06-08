import unittest
from textflint.input_layer.model.metrics.metrics import *
import numpy as np


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

    def test_pos_metric(self):
        pred = np.array([[1, 2, 1], [2, 1, 1]])
        gold = np.array([[1, 2, 0], [1, 1, 0]])
        pos_metric = POSMetric()
        score = pos_metric(pred, gold, ignore_label_id=0)
        self.assertEqual(0.75, score['precision'])


if __name__ == '__main__':
    unittest.main()
