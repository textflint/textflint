r"""
Concatenate sentences to a longer one.
==========================================================
"""
__all__ = ["ConcatSent"]
from ...transformation import Transformation


class ConcatSent(Transformation):
    r"""
    Concatenate sentences to a longer one.Generate one data sample at most.

    """
    def __init__(
        self,
        **kwargs
    ):
        r"""
        :param string res_path: dir for vocab/dict
        :param **kwargs:
        """
        super().__init__(**kwargs)

    def __repr__(self):
        return 'ConcatCase'

    def _transform(self, input_sample, n=1, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~NERSample input_sample: Data sample for augmentation
        :param int n: Default is 1. MAx number of unique augmented output
        :param **kwargs:
        :return: Augmented NERSample.
        """
        concat_samples = kwargs.get('concat_samples', [])

        if not concat_samples:
            return []

        ori_samples = [input_sample] + concat_samples

        word_sequences = []
        tag_sequences = []
        sample = input_sample.clone(input_sample)

        for ori_sample in ori_samples:
            word_sequences.extend(ori_sample.text.tokens)
            tag_sequences.extend(ori_sample.tags)

        sample = sample.unequal_replace_field_at_indices('text', [(0, len(input_sample.text))], [word_sequences])
        sample = sample.unequal_replace_field_at_indices('tags', [(0, len(input_sample.text))], [tag_sequences])

        sample.entities = sample.find_entities_BMOES(sample.text.tokens, sample.tags)

        return [sample]
