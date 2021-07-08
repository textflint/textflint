r"""
Generate some samples by templates
       implement follow
       Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference ACL2019
       In order to generate some sample whose premise is the sequence of the hypothesis but the semantic are different.
==========================================================
"""

from ..transformation import Transformation
from ....common.utils.overlap_templates import *
from ....input.component.sample import NLISample

__all__ = ['Overlap']


def no_the(sentence):
    return sentence.replace("the ", "")


def repeaters(sentence):
    condensed = no_the(sentence)
    words = []

    for word in condensed.split():
        if word in lemma:
            words.append(lemma[word])
        else:
            words.append(word)

    if len(list(set(words))) == len(words):
        return False
    else:
        return True


class Overlap(Transformation):
    r"""
    Generate some samples by templates
   implement follow
   Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural
   Language Inference ACL2019
   In order to generate some sample whose premise is the sequence of the
   hypothesis but the semantic are different.
   exmaple:
   {
        hypothesis: I hope Tom can go to school.
        premise: Tom goes to school.
        y: contradicion
   }
    """
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'Overlap'

    def transform(self, sample, n=1, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~NLISample sample: Data sample for augmentation
        :param int n: Default is 1. MAX number of unique augmented output
        :param **kwargs:
        :return: Augmented data
        """
        transform_results = self._transform(n, **kwargs)

        if transform_results:
            return transform_results
        else:
            return []

    def _transform(self, n=5, **kwargs):
        r"""
        Transform text string, this kind of transformation can only produce
        one sample.

        :param ~NLISample sample: input data, a NLISample contains 'hypothesis'
            field, 'premise' field and 'y' field
        :param int n: number of generated samples, this transformation can only
            generate one sample
        :return list trans_samples: transformed sample list that only contain
            one sample
        """

        example_counter = 0
        trans_list = []
        for template_tuple in template_list:
            label = template_tuple[2]
            template = template_tuple[3]

            example_dict = {}
            count_examples = 0

            while count_examples < n:
                example = template_filler(template)

                example_sents = tuple(example[:2])

                if example_sents not in example_dict and not repeaters(example[0]):
                    example_dict[example_sents] = 1
                    trans_sample = {
                        'hypothesis': example[0],
                        'premise': example[1],
                        'y': label
                    }
                    trans_list.append(NLISample(trans_sample))
                    count_examples += 1
                    example_counter += 1

        return trans_list

