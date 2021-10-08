r"""
WSD - SwapTarget_syn: For one sample, replace the target word with its synonym in wordnet
==========================================================
"""
import random
from ...transformation import Transformation
from ....input.component.sample.wsd_sample import WSDSample
from nltk.corpus import wordnet as wn

__all__ = ['SwapTarget']


class SwapTarget(Transformation):
    r"""
    replace the target word with its synonym in wordnet
    """

    def __init__(
            self,
            replacement_type='syn',
            **kwargs
    ):
        r"""
        :param replacement_type: case type, only support
            ['syn']
        """
        super().__init__()
        if replacement_type not in ['syn']:
            raise ValueError(
                'Not support {0} type, plz ensure replacement_type in {1}'.format(
                    replacement_type, ['syn']))
        self.replacement_type = replacement_type

    def __repr__(self):
        return 'SwapTarget' + '_' + self.replacement_type

    def _transform(self, sample, n=1, field='sentence', **kwargs):
        r"""
        :param ~textflint.WSDSample sample: a WSDSample
        :param int n: optional; number of generated samples
        :param str|list fields: field to transform
        :return list: new, transformed sample list.

        """

        def wn_sensekey2synset(sensekey):
            r"""
            Convert sensekey to synset.
            :param str sensekey: sense key according to wordnet
            :return synset: synset including this sense key
            :return lemma: lemma extracted from this sense key
            """
            lemma = sensekey.split('%')[0]
            for synset in wn.synsets(lemma):
                for lemma in synset.lemmas():
                    if lemma.key() == sensekey:
                        return synset, lemma
            return None, None

        assert (isinstance(sample, WSDSample)), \
            f"the type of 'sample' should be WSDSample, got " \
            f"{type(sample)} instead"
        # don't support this transformation
        instance = sample.instance
        idx_list = list()
        sk_list = list()
        flag = 0
        for key, start, end, word, sk in instance:
            syn, lem = wn_sensekey2synset(sk)
            # skip this instance
            if syn is None:
                continue
            lem_list = list()
            for l in syn.lemmas():
                if l == lem:
                    continue
                lem_list.append(l)
            # skip this instance
            if len(lem_list) == 0:
                continue
            flag = 1  # this sample can be transformed
            idx = random.randint(0, len(lem_list) - 1)
            rep_lem = lem_list[idx]
            new_sk = rep_lem.key()
            idx_list.append([start, end])
            sk_list.append([new_sk])
        # skip this sample
        if flag == 0:
            return []
        # replace with new sense key
        transformed_sample = sample.unequal_replace_field_at_indices('sentence',
                                                                     idx_list,
                                                                     sk_list)
        return [transformed_sample]
