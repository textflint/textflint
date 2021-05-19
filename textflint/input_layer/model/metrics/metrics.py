import torch
import datasets
from sklearn.metrics import accuracy_score, \
    precision_score, \
    recall_score, \
    f1_score


class SQuADMetric(object):
    def __init__(self):
        self.metric = datasets.load_metric("squad")

    def __call__(self, predictions, references):
        r"""
        Computes SQuAD scores (F1 and EM).
        :param list predictions:
        List of question-answers dictionaries with the following key-values:
            - 'id': id of the question-answer pair as given in the references
            - 'prediction_text': the text of the answer
        :param list references:
        List of question-answers dictionaries with the following key-values:
            - 'id': id of the question-answer pair (see above),
            - 'answers': a Dict in the SQuAD dataset format
                {
                    'text': list of possible texts for the answer
                    'answer_start': list of start positions for the answer
                }
        :return: dict obj to save metrics result
        """
        return self.metric.compute(
            predictions=predictions, references=references)
