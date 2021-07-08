r"""
Extract samples with long text or short text
============================================

"""
__all__ = ['LengthSubPopulation']
from ..subpopulation import SubPopulation


class LengthSubPopulation(SubPopulation):
    r"""
    Filter samples based on text length

    Example::

        sample 1: "I love textflint", score: 3
        sample 2: "I love textflint very much", score: 5
    """
    def __init__(
            self,
            intervals=["0%", "20%"]
    ):
        if intervals is None:
            raise ValueError(
                'Intervals should be initialized for LengthSubPopulation')
        super().__init__(intervals=intervals)

    def __repr__(self):
        return "LengthSubPopulation-" + \
            str(self.intervals[0]) + "-" + str(self.intervals[1])

    def _score(self, sample, fields, **kwargs):
        r"""
        Calculate the score based on text length

        :param sample: data sample
        :param list fields: list of field str
        :param kwargs:
        :return int: score for sample

        """
        words = []
        for field in fields:
            words += sample.get_words(field)
        return len(words)
