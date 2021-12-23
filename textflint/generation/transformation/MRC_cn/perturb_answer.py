r"""
Perturb Answer by altering the sentence that contains answer
==========================================================
"""

import collections
from copy import deepcopy

from ....common.settings import ORIGIN
from ..transformation import Transformation
from ..UT_cn import CnSwapSynWordEmbedding
from ....input.component.sample import MRCCnSample,UTCnSample
from ....common.settings import CN_EMBEDDING_PATH
from ....common.utils.load import json_lines_loader
from ....common.utils.install import download_if_needed

class PerturbAnswer(Transformation):
    r"""
    Transform the sentence containing answer with AlterSentence transformation.

    Example::

        origin sentence: Denver Broncos defeated the National Football
            Conference champion Carolina Panthers 24–10 to earn their
            third Super Bowl title.
        transformed sentence: Denver Broncos defeated the National Football
            Conference champ Carolina Panthers 24–10 to earn
            their 3rd Super Bowl rubric.
    """

    def __init__(self):
        super().__init__()
        self.sim_dic = json_lines_loader(download_if_needed(CN_EMBEDDING_PATH))[0]

    def __repr__(self):
        return 'PerturbAnswer'

    def _transform(
            self,
            sample,
            nearby_word_dict=None,
            pos_tag_dict=None,
            **kwargs
    ):
        r"""
        Extract the sentence with answer from context, replace synonyms
            based on WordNet and glove
        embedding space while keep the semantic meaning unchanged.
        :param sample: the sample to transform
        :param dict nearby_word_dict: the dict to search for nearby words
        :param dict pos_tag_dict: the dict to search for
            the most frequent pos tags
        :param kwargs:
        :return: list of sample
        """

        # filter no-answer samples
        if sample.is_impossible:
            return []
        answers = sample.get_answers()
        answer_token_start = answers[0]['start']
        answer_token_end = answers[0]['end']
        answer_text = answers[0]['text']
        sentences = sample.get_sentences('context')

        sent_start = 0
        alter_sent = None
        indices = None
        # Pick up the sentence that contains the answer
        for i, sent in enumerate(sentences):
            if sent_start + len(self.cn_processor.tokenize(sent,cws=False)) \
                    <= answer_token_start:
                sent_start += len(self.cn_processor.tokenize(sent,cws=False))
                continue
            # deal with sentence tokenize error
            if sent.find(answer_text) < 0:
                return []
            # Transform a sentence with AlterSentence function
            alter_sent, indices = self.alter_sentence(sent)

            indices = [index + sent_start for index in indices]
            break
        if alter_sent is None:
            return None
        transform_samples = []
        results = []
        replace_items = []
        words = self.cn_processor.tokenize(alter_sent,cws = False)
        context_mask = sample.get_mask('context')

        for index in indices:
            if index >= len(context_mask):
                return []
            if context_mask[index] != ORIGIN:
                continue
            results.append(index)
            replace_items.append(words[index - sent_start])
        if results:
            new_sample = sample.replace_field_at_indices(
                'context', results, replace_items)
        else:
            return []
        transform_samples.append(new_sample)

        return transform_samples

    def alter_sentence(self,sent):
        sent = deepcopy(sent)
        words = self.cn_processor.tokenize(sent)
        cnt = 0
        for word in words:
            if cnt > 4:
                break
            synwords  = self.word_in_sim_dic(word)
            if synwords and len(word) == len(synwords[0]):
                cnt += 1
                sent = sent.replace(word,synwords[0])
        indices = range(0,len(sent))
        return sent,indices

    def word_in_sim_dic(self, word):
        if word in self.sim_dic:
            return self.sim_dic[word]
        else:
            return []