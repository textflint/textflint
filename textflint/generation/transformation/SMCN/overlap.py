r"""
Generate some samples by templates
       implement follow
       Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference ACL2019
       In order to generate some sample whose premise is the sequence of the hypothesis but the semantic are different.
==========================================================
"""
__all__ = ['Overlap']
from textflint.generation.transformation import Transformation
from textflint.common.utils.c_overlap_templates import *
from textflint.input.component.sample import SMCNSample


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
    Generate some samples by templates which implement follow
        Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural
        Language Inference ACL2019

       In order to generate some sample whose premise is the sequence of the
       hypothesis but the semantic are different.

    Exmaple::

       {
            sentence1: 我以为他去上学了。
            sentence2: 他去上学了。
            y: 0
       }

    """

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'Overlap'

    def transform(self, sample, n=1, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~SMCNSample sample: Data sample for augmentation
        :param int n: Default is 1. MAX number of unique augmented output
        :param **kwargs:
        :return: Augmented data

        """
        transform_results = self._transform(n, **kwargs)

        if transform_results:
            return transform_results
        else:
            return []

    def _transform(self, n=1, **kwargs):
        r"""
        Transform text string, this kind of transformation can only produce
        one sample.

        :param ~SMCNSample sample: input data, a NLISample contains
            'sentence1' field, 'sentence2' field and 'y' field
        :param int n: number of generated samples, this transformation can
            only generate one sample
        :return list trans_samples: transformed sample list that only
            contain one sample

        """
        example_counter = 0
        trans_list = []

        for template_tuple in template_list:
            label = template_tuple[2]
            if label == 'entailment':
                label = '1'
            else:
                label = '0'

            template = template_tuple[3]
            example_dict = {}
            count_examples = 0

            while count_examples < n:
                example = template_filler(template)
                flag = example[2]
                if_repeat = repeaters(example[0])
                if flag == 'temp53' or flag == 'temp54':
                    if_repeat = False
                example_sents = tuple(example[:2])

                if example_sents not in example_dict and not if_repeat:
                    example_dict[example_sents] = 1
                    trans_sample = {
                        'sentence1': example[0],
                        'sentence2': example[1],
                        'y': label
                    }

                    trans_list.append(SMCNSample(trans_sample))
                    count_examples += 1
                    example_counter += 1

        return trans_list


