r"""
SwapMultiPOS transformation for POS tagging
==========================================================
"""
__all__ = ["SwapMultiPOS"]

from ....input.component.sample import POSSample
from ...transformation import WordSubstitute


class SwapMultiPOS(WordSubstitute):
    r"""
    Word Swap by swaping words that have multiple POS tags in WordNet.

    """
    def __init__(self,
                 treebank_tag="JJ",
                 trans_max=2,
                 trans_p=1,
                 **kwargs
                 ):
        r"""

        :param treebank_tag: words with this pos tag will be replaced
        :param kwargs:
        """
        super().__init__(trans_max, trans_p, **kwargs)
        self.treebank_tag = treebank_tag
        self.check_pos()
        self.wordnet_candidates = self.get_candidates_from_wordnet()
        self.get_pos = True

    def __repr__(self):
        return 'SwapMultiPOS' + '-' + self.treebank_tag

    def check_pos(self):
        support_pos = ["JJ", "NN", "VB", "RB"]
        assert self.treebank_tag in support_pos, \
            "Only support replacing JJ, NN, VB and RB!"

    def get_candidates_from_wordnet(self):
        r"""
        get all possible multi-pos words with pos tags same as treebank_tag.

        :return: a list
        """
        noun = set(
            [i for i in self.processor.get_all_lemmas(pos='n') if "_" not in i])
        verb = set(
            [i for i in self.processor.get_all_lemmas(pos='v') if "_" not in i])
        adj = set(
            [i for i in self.processor.get_all_lemmas(pos='a') if "_" not in i])
        adv = set(
            [i for i in self.processor.get_all_lemmas(pos='r') if "_" not in i])

        candidates = []
        if self.treebank_tag == "NN":
            candidates = list(noun & (verb | adj | adv))
        elif self.treebank_tag == "VB":
            candidates = list(verb & (noun | adj | adv))
        elif self.treebank_tag == "JJ":
            candidates = list(adj & (verb | noun | adv))
        elif self.treebank_tag == "RB":
            candidates = list(adv & (verb | adj | noun))
        return candidates

    def _get_candidates(self, word, pos=None, n=5):
        r"""
        Returns a list containing all possible words.

        :param word: str, the word to replace
        :param pos: str, the pos of the word to replace
        :param n: the number of returned words
        :return: a candidates list
        """
        return self.sample_num(self.wordnet_candidates, n)

    def skip_aug(self, tokens, mask, pos=None):
        r"""
        Returns the index of the replaced tokens.

        :param tokens: list, tokenized words or word with pos tag pairs
        :param mask: list, the mask symbol of the tokens
        :param pos: list, the pos tags of the tokens
        :return: list, the words at these indices that can be replaced
        """
        assert pos is not None, "POS tag must be given!"
        results = []
        indices = self.pre_skip_aug(tokens, mask)

        for index in indices:
            if pos[index] == self.treebank_tag:
                results.append(index)
        return results
