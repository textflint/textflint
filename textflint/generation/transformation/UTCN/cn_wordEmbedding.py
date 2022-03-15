r"""
Swapping words by Glove
==========================================================
"""

__all__ = ['CnSwapSynWordEmbedding']

from ...transformation import CnWordSubstitute
from ....common.settings import CN_EMBEDDING_PATH
from ....common.utils.load import json_lines_loader
from ....common.utils.install import download_if_needed


class CnSwapSynWordEmbedding(CnWordSubstitute):
    r"""
    Transforms an input by replacing its words by Glove.
    Offline Vocabulary contains top 17433 common words and its top 20 synonym.

    """
    def __init__(
        self,
        trans_min=1,
        trans_max=10,
        trans_p=0.1,
        stop_words=None,
        islist=False,
        **kwargs
    ):
        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words
        )
        self.get_pos = True
        self.islist = islist
        self.sim_dic = json_lines_loader(download_if_needed(CN_EMBEDDING_PATH))[0]

    def __repr__(self):
        return 'CnSwapSynWordEmbedding'

    def _get_candidates(self, word, pos=None, n=5):
        r"""
        Returns a list containing all possible words with 1 character replaced
        by word embedding.

        :param str word: token word to transform.
        :param str pos:
        :param int n: max number of candidates
        :return : candidates list

        """
        sim_list = self.word_in_sim_dic(word)

        return self.sample_num(sim_list, n)

    def word_in_sim_dic(self, word):
        if word in self.sim_dic:
            return self.sim_dic[word]
        else:
            return []

    def skip_aug(self, words, words_indices, tokens, mask, **kwargs):
        if self.islist:
            return self.pre_skip_aug_list(words, words_indices, tokens, mask)
        else:
            return self.pre_skip_aug(words, words_indices, tokens, mask)
