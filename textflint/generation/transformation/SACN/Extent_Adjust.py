r"""
change sentiment extent by adding adverb
==========================================================

"""
__all__ = ["ExtentAdjust"]
from ..transformation import Transformation
from ....common.settings import CNSA_EXTENT_LIST
from random import choice


class ExtentAdjust(Transformation):
    r"""
    Transforms an input by change its sentiment extent by adding adverb

    Example::

        ori: 卫生间不错，好用
        trans: 卫生间比较不错，有点好用
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.extent_list = CNSA_EXTENT_LIST

    def __repr__(self):
        return 'ExtentAdjust'

    def _transform(self, sample, n=1, **kwargs):
        r"""
        Transform text string according field, this kind of transformation
        can only produce one sample.

        :param ~SASample sample: input data, a SASample contains 'x' field
            and 'y' field
        :param int n: number of generated samples, this transformation can
            only generate one sample
        :return list trans_samples: transformed sample list that only contain
            one sample

        """
        tags = sample.get_pos('x')
        a_tags = [tag for tag in tags if tag[0] == 'a']
        if not a_tags:
            return []

        a_tags.sort(key=lambda x: x[1], reverse=True)
        for a_tag in a_tags:
            _,  f_loc, _ = a_tag
            adverb = choice(self.extent_list)
            sample = sample.insert_field_before_index('x', f_loc, adverb)

        return [sample]

