import json

from ..metrics.nerspanmetric import NERSpanMetric
from .flint_model import TASK_METRICS, FlintModel
from ..test_model.bilstm_crf_torch import TaggerFactory


class FlintModelNER(FlintModel):
    def __init__(self,
                 model,
                 batch_size=100,
                 metrics=None,
                 encoding_type='bio'
                 ):
        r"""

        :param model: any model object
        :param int batch_size: batch size to apply evaluation
        :param Object metrics: metrics takes the predicted labels
        and the ground-truth labels as input and calculates f1, recall
        and precision. It takes a NERSpanMetric metric
        as default, but can also be implemented by the user.
        :param str encoding_type: supporting 'bio', 'bmes', 'bmeso', 'bioes',
        'bio' as default.
        """
        super(FlintModelNER, self).__init__(
            model=model,
            tokenizer=None,
            task='NER',
            batch_size=batch_size)
        # load augments for the model
        self.batch_size = batch_size
        self.model = model
        self.encoding_type = encoding_type
        self.metrics2score = {'f': 'f1_score',
                              'pre': 'precision', 'rec': 'recall'}
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = TASK_METRICS['NER'][0]['fun'](
                encoding_type=self.encoding_type
            )

    def evaluate(self, data_samples, prefix=""):
        r"""
        :param list[Sample] data_samples: list of Samples
        :return: dict obj to save metrics result

        """
        word_seq, target_tag_seq, seq_len = self.encode(data_samples)
        pred_tag_seq = self.model.predict_tags_from_words(
            word_seq,
            batch_size=self.batch_size
        )
        metric = self.metrics
        metric.evaluate(
            pred_tag_seq,
            target_tag_seq,
            seq_len
        )
        score = {}
        for key in metric.get_metric():
            score[prefix + self.metrics2score[key]] = metric.get_metric()[key]
        return score

    def encode(self, inputs):
        r"""
        NER evaluator for data preprocess
        :param list[Sample] data_samples: list of Samples
        :return: DataSet obj obtaining

        """
        word_seq = []
        tag_seq = []
        seq_len = []
        for sample in inputs:
            word_seq.append(sample['x'])
            tag_seq.append(sample['y'])
            seq_len.append(len(sample['x']))
        return word_seq, tag_seq, seq_len

