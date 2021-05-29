import torch
from fastNLP import SpanFPreRecMetric, DataSet, DataSetIter, SequentialSampler, Vocabulary
from fastNLP.core.utils import _move_dict_value_to_device
from fastNLP.io import DataBundle
from fastNLP.io.pipe.conll import _NERPipe

from ..test_model.bilstm_crf_torch import BilstmCRFTorch, device
from .flint_model import TASK_METRICS, FlintModel


class FlintModelNer(FlintModel):
    def __init__(self,
                 model,
                 batch_size=32,
                 metrics=None,
                 tag_vocab=None,
                 word_vocab=None,
                 encoding_type='bio'
                 ):
        r"""

        :param model: any model object
        :param int batch_size: batch size to apply evaluation
        :param Object metrics: metrics takes the predicted labels
        and the ground-truth labels as input and calculates f1, recall
        and precision. It takes a SpanFPreRecMetric(in FastNLP) metric
        as default, but can also be implemented by the user.
        :param list/dict/Vocabulary(in FastNLP) tag_vocab: a dict or Vocabulary
        for mapping tags to indexes.
        :param list/dict/Vocabulary(in FastNLP) word_vocab: a dict or Vocabulary
        for mapping words to indexes.
        :param str encoding_type: supporting 'bio', 'bmes', 'bmeso', 'bioes',
        'bio' as default.
        """
        super(FlintModelNer, self).__init__(
            model=model,
            tokenizer=None,
            task='NER',
            batch_size=batch_size)
        # load augments for the model
        self.batch_size = batch_size
        self.model = model
        self.encoding_type = encoding_type
        self.tag_vocab = self.get_vocab(tag_vocab)
        self.word_vocab = self.get_vocab(word_vocab)
        self.metrics2score = {'f': 'f1_score',
                              'pre': 'precision', 'rec': 'recall'}
        if metrics is not None:
            self.metrics = metrics
        else:
            self.metrics = TASK_METRICS['NER'][0]['fun'](
                tag_vocab=self.tag_vocab,
                encoding_type=self.encoding_type)

    def evaluate(self, data_samples, prefix=""):
        r"""
        :param list[Sample] data_samples: list of Samples
        :return: dict obj to save metrics result

        """
        dataset = self.encode(data_samples)
        data_iterator = DataSetIter(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=SequentialSampler())
        with torch.no_grad():
            for batch_x, batch_y in data_iterator:
                _move_dict_value_to_device(batch_x, batch_y, device=device)
                pred_dict = self.model.forward(
                    words=batch_x['words'],
                    seq_len=batch_x['seq_len'])
                pred = pred_dict['pred'] # [batch, seq_len]
                target = batch_y['target'] # [batch, seq_len]
                seq_len = batch_x['seq_len'] # [batch]
                self.metrics.evaluate(pred, target, seq_len)

        metric = self.metrics.get_metric()
        score = {}
        for i in metric.keys():
            score[prefix + self.metrics2score[i]] = metric[i]
        return score

    def get_vocab(self, my_vocab):
        r"""
        turn a list or dict to a Vocabulary.
        for a dict input, the format is like {'B-PER': 0, "I-PER": 1, ...}
        for a list input, make sure that all indexes of labels/words is
        compatible with the indexes the model uses.
        :param list/dict/Vocabulary dic: list/dict of tokens
        :return: Vocabulary obj to save the vocab

        """
        if isinstance(my_vocab, Vocabulary):
            return my_vocab
        vocab=Vocabulary(unknown=None, padding=None)
        if isinstance(my_vocab, dict):
            my_vocab = sorted(my_vocab.items(), key=lambda item:item[1])
            my_vocab = [word[0] for word in my_vocab]
        assert isinstance(my_vocab, list), "You should input a list or dict!"
        vocab.add_word_lst(my_vocab)
        return vocab

    def encode(self, inputs):
        r"""
        NER evaluator for data preprocess
        :param list[Sample] data_samples: list of Samples
        :return: DataSet obj obtaining

        """
        dataset = self.reformat(inputs)
        data_b = DataBundle(vocabs={'words': self.word_vocab,
                                    "target": self.tag_vocab},
                            datasets={'test': dataset})
        data_b = _NERPipe(encoding_type=self.encoding_type).process(data_b)

        return data_b.get_dataset('test')

    def reformat(self, inputs):
        r"""
        :param list[Sample] inputs: list of Samples
        :return: DataSet obj obtaining

        """
        raw_words = []
        target = []
        seq_len = []
        for input in inputs:
            sent = input['x']
            tags = input['y']
            length = len(input['y'])
            raw_words.append(sent)
            target.append(tags)
            seq_len.append(length)
        data = {'raw_words': raw_words,
                'target': target,
                "seq_len": seq_len}
        dataset = DataSet(data)
        return dataset

