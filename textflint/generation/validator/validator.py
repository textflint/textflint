r"""
Constraint Class
=====================================
"""
import numpy
from abc import ABC, abstractmethod

from ...input.dataset.dataset import *
__all__ = ['Validator']


class Validator(ABC):
    r"""
    An abstract class that computes the semantic similarity score between
        original text and adversarial texts

    :param ~textflint.input.dataset origin_dataset:
            the dataset of origin sample
    :param ~textflint.input.dataset trans_dataset:
        the dataset of translate sample
    :param str|list fields: the name of the origin field need compare.
    :param bool need_tokens: if we need tokenize the sentence

    """
    def __init__(
        self,
        origin_dataset,
        trans_dataset,
        fields,
        need_tokens=False
    ):
        assert isinstance(origin_dataset, Dataset), f"Input must be a {Dataset}"
        assert isinstance(trans_dataset, Dataset), f"Input must be a {Dataset}"
        assert len(origin_dataset) and len(trans_dataset), \
            f"origin_dataset and trans_dataset can not be empty."
        assert isinstance(origin_dataset[0], type(trans_dataset[0])), \
            f"The type of origin sample and trans sample must be same."

        self.ori_dataset = origin_dataset
        self.trans_dataset = trans_dataset
        self.id2loc = None
        self._score = None
        self.fields = fields if isinstance(fields, list) else [fields]
        self.check_data()
        self.need_tokens = need_tokens

    @abstractmethod
    def validate(self, transformed_text, reference_text):
        r"""
        Calculate the score

        :param str transformed_text: transformed sentence
        :param str reference_text: origin sentence
        :return float: the score of two sentence

        """
        raise NotImplementedError()

    def check_data(self):
        r"""
        Check whether the input data is legal

        """
        self.id2loc = {}
        for i in range(len(self.ori_dataset)):
            if self.ori_dataset[i].sample_id not in self.id2loc:
                self.id2loc[self.ori_dataset[i].sample_id] = i
            else:
                raise ValueError('There are two origin samples have same id.')
            if not isinstance(self.ori_dataset[0], type(self.ori_dataset[i])):
                raise ValueError('There are at least two type of '
                                 'origin sample in the dataset.')
            try:
                for field in self.fields:
                    self.ori_dataset[i].get_value(field)
            except AttributeError:
                raise ValueError(f'The {i} sample does not '
                                 f'have the attribute {field}.')

        for i, trans_data in enumerate(self.trans_dataset):
            if trans_data.sample_id not in self.id2loc:
                raise ValueError('There is no origin sample '
                                 'can match trans sample')
            if not isinstance(self.trans_dataset[0], type(trans_data)):
                raise ValueError('There are at least two type of '
                                 'origin sample in the dataset.')
            try:
                for field in self.fields:
                    trans_data.get_value(field)
            except AttributeError:
                raise ValueError(f'The {i} sample does not '
                                 f'have the attribute {field}.')

    @property
    def score(self):
        r"""
        Calculate the score of the deformed sentence

        :return list: a list of translate sentence score

        """
        if not self._score:
            self.check_data()
            self._score = []

            for trans_sample in self.trans_dataset:
                for field in self.fields:
                    score = []
                    trans = trans_sample.get_words(field) \
                        if self.need_tokens else trans_sample.get_text(field)

                    ori = self.ori_dataset[self.id2loc[trans_sample.sample_id]]
                    ori = ori.get_words(field) if self.need_tokens \
                        else ori.get_text(field)
                    score.append(self.validate(trans, ori))

                self._score.append(numpy.mean(score))
            assert len(self._score) == len(self.trans_dataset), \
                "The len of the score not equal transset."

        return self._score


