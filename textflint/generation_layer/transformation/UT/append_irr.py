r"""
extend sentences by irrelevant sentences
==========================================================
"""

__all__ = ['AppendIrr']

from ..transformation import Transformation
from ....common.utils.load import plain_lines_loader
from ....common.utils.install import download_if_needed
from ....common.settings import MIN_SENT_TRANS_LENGTH, BEGINNING_PATH, \
    PROVERB_PATH


class AppendIrr(Transformation):
    r"""
    Extend sentences by irrelevant sentences.

    Example::

        given "input sentence"
    ->  beginning + "input sentence" + proverb

    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.beginnings = plain_lines_loader(
            download_if_needed(BEGINNING_PATH))
        self.proverbs = plain_lines_loader(download_if_needed(PROVERB_PATH))

    def __repr__(self):
        return 'AppendIrr'

    def _transform(self, sample, field='x', n=5, **kwargs):
        r"""
        Transform text string according transform_field.

        :param ~Sample sample:  input data, normally one data component.
        :param str field: indicate which field to transform
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.
        """

        trans_samples = []
        tokens = sample.get_words(field)
        beginnings = self._get_beginnings(n)

        # add beginning phrase
        for i in range(min(n, len(beginnings))):
            trans_samples.append(
                sample.insert_field_before_index(
                    field, 0, beginnings[i]))

        if len(tokens) < MIN_SENT_TRANS_LENGTH:
            return trans_samples

        # add proverb to the end
        proverbs = self._get_proverbs(n)

        for i in range(min(len(trans_samples), len(proverbs))):
            _tokens = trans_samples[i].get_words(field)
            trans_samples[i] = trans_samples[i].insert_field_after_index(
                field, len(_tokens) - 1, proverbs[i])

        return trans_samples

    def _get_beginnings(self, n):
        beginnings = self.sample_num(self.beginnings, n)

        return [self.processor.tokenize(
            beginning) for beginning in beginnings]

    def _get_proverbs(self, n):
        proverbs = self.sample_num(self.proverbs, n)

        return [self.processor.tokenize(proverb) for proverb in proverbs]

