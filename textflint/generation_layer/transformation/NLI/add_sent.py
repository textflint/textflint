r"""
Adding the Meaningless sentences to the hypothesis
==========================================================
"""

from ..transformation import Transformation

__all__ = ['AddSent']


class AddSent(Transformation):
    r"""
    Adding the Meaningless sentences to the hypothesis and remain both premise
    and label.
    Users can use their own meaningless sentences by change the text in
    transform method

    exmaple:
    {
        hypothesis: I hate this book.
        premise: This book is my favorite. today is Monday.
        y: contradiction
    }
    """

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'AddSent'

    # TODO, add irrelevant sentence resource
    def transform(
            self,
            sample,
            n=1,
            text=' And we have to say that the sun '
                 'is not moon, the moon is not sun.',
            **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~NLISample sample: Data sample for augmentation
        :param int n: Default is 1. MAX number of unique augmented output
        :param **kwargs:
        :return: Augmented data
        """
        transform_results = self._transform(sample, text, n=1, **kwargs)

        if transform_results:
            return [data for data in transform_results if not data.is_origin]
        else:
            return []

    def _transform(self, sample, text, n=1, **kwargs):
        r"""
        Transform text string, this kind of transformation can only produce one
        sample.

        :param ~NLISample sample: input data, a NLISample contains 'hypothesis'
            field, 'premise' field and 'y' field
        :param int n: number of generated samples, this transformation can only
            generate one sample
        :return list trans_samples: transformed sample list that only contain
            one sample
        """

        label_tag = sample.get_value('y')
        original_text1 = sample.get_text('hypothesis') + text
        original_text2 = sample.get_text('premise')

        sample = sample.replace_fields(['hypothesis', 'premise', 'y'],
                                       [original_text1, original_text2,
                                        label_tag])

        return [sample]
