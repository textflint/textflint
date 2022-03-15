r"""
order swap by swapping sub-sentence to different orders
==========================================================
"""
__all__ = ["SentenceOrderSwap"]
from ..transformation import Transformation
import random


class SentenceOrderSwap(Transformation):
    r"""
    order swap by swapping sub-sentence to different orders。

    Example::

        ori: 拿房时没大床房了，给我们免费升成套房，这点还蛮满意的。酒店大致不错，有国内五星水准。
            比国际品牌的要差一点。酒店有点年纪了，维修要加强，比如我们浴室的下水就堵塞不通，这些在客人入住前就该发觉修好。其它都还可以。

        trans: 比国际品牌的要差一点。维修要加强，其它都还可以。比如我们浴室的下水就堵塞不通，
            给我们免费升成套房，拿房时没大床房了，酒店大致不错，有国内五星水准，这点还蛮满意的。酒店有点年纪了，这些在客人入住前就该发觉修好。
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

    def __repr__(self):
        return 'SentenceOrderSwap'

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
        sentences = sample.get_value('x')
        sub_sentences = sample.get_sentences('x')
        sentence_len = len(sentences)

        if not sub_sentences:
            return []

        random.shuffle(sub_sentences)
        swap_sentence = ''.join(sub_sentences)
        sample = sample.delete_field_at_indices('x', [0, sentence_len-1])
        sample = sample.insert_field_before_index('x', 0, swap_sentence)
        return [sample]
