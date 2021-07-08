r"""
SubPopulation Abstract Class
============================================

"""

import math
from tqdm import tqdm
from abc import ABC, abstractmethod

from ...common.preprocess import EnProcessor


class SubPopulation(ABC):
    r"""
    An abstract class for extracting subset of examples.

    """
    text_processor = EnProcessor()

    def __init__(
        self,
        intervals=None,
        **kwargs
    ):
        self.intervals = intervals

    def __repr__(self):
        return "SubPopulation"

    def score(self, sample, field, **kwargs):
        r"""
        Score the sample

        :param sample: data sample
        :param str|list field: field str
        :param kwargs:
        :return int: score for sample

        """

        if not isinstance(field, list):
            fields = [field]
        else:
            fields = field
        return self._score(sample, fields, **kwargs)

    @abstractmethod
    def _score(self, samle, fields, **kwargs):
        r"""
        Score the sample

        :param sample: data sample
        :param list fields: list of field str
        :param kwargs:
        :return int: score for sample

        """
        raise NotImplementedError

    def get_slice(self, scores, dataset):
        r"""
        Pick up samples based on scores

        :param list scores: list of int
        :param dataset: Dataset
        :return: subset samples

        """
        indexes = sorted(range(len(scores)), key=lambda x: scores[x])
        sort_samples = [dataset[idx] for idx in indexes]
        left_bound = self.normalize_bound(self.intervals[0], len(dataset))
        right_bound = self.normalize_bound(self.intervals[1], len(dataset))
        sub_samples = sort_samples[left_bound:right_bound]

        return sub_samples

    def slice_population(self, dataset, fields, **kwargs):
        r"""
        Extract a subset of samples.

        :param dataset: Dataset
        :param list fields: field str list
        :param kwargs:
        :return: Subset Dataset

        """

        scores = []
        for sample in tqdm(dataset):
            scores.append(self.score(sample, fields))
        sub_samples = self.get_slice(scores, dataset)
        new_dataset = dataset.new_dataset()
        new_dataset.extend(sub_samples)
        return new_dataset

    @staticmethod
    def normalize_bound(limit, size):
        r"""
        Normalize the bound of slice

        :param str|float|int limit: left_bound or right_bound for intervals
            can be percentile like 10%, 20%
            can be float between 0 and 1 like 0.3
            can be int index like 50
        :param size: the size of samples
        :return int : bound

        """
        if isinstance(limit, str) and limit.endswith("%"):
            limit = float(limit.replace("%", "")) / 100
            return math.floor(limit * size)
        elif isinstance(limit, float):
            return math.floor(limit * size)
        elif isinstance(limit, int):
            return limit
        else:
            raise NotImplementedError
