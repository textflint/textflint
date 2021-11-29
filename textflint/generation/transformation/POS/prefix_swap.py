r"""
SwapPrefix transformation for POS tagging
============================================
"""
__all__ = ["SwapPrefix"]

from collections import Counter
from ....input.component.sample import POSSample
from ...transformation import WordSubstitute
from ....common.utils.load import load_morfessor_model
from ....common.utils.install import download_if_needed
from ....common.settings import MORPHEME_ANALYZER


class SwapPrefix(WordSubstitute):
    r"""
    Swap prefix and keep the same POS tags.

    """
    def __init__(self,
                 trans_max=2,
                 trans_p=1,
                 **kwargs
                 ):
        super().__init__(trans_max, trans_p, **kwargs)
        self.morpheme_analyzer = load_morfessor_model(
            download_if_needed(MORPHEME_ANALYZER))
        self.remain_prefix_dict = self.get_remain_prefix_dict()
        self.get_pos = True

    def __repr__(self):
        return 'SwapPrefix'

    def get_remain_prefix_dict(self):
        r"""
        Get all possible candidates from WordNet.

        :return: a dict used as inverted index, {remain: prefix},
        """
        dicts = {
            'NN': [
                i for i in self.processor.get_all_lemmas(
                    pos='n') if "_" not in i], 'VB': [
                i for i in self.processor.get_all_lemmas(
                    pos='v') if "_" not in i], 'JJ': [
                        i for i in self.processor.get_all_lemmas(
                            pos='a') if "_" not in i], 'RB': [
                                i for i in self.processor.get_all_lemmas(
                                    pos='r') if "_" not in i]}
        remain_prefix_dict = {}
        prefix_counter = Counter()
        for k, v in dicts.items():
            for w in v:
                segs, _ = self.morpheme_analyzer.viterbi_segment(w)

                if len(segs) > 1:
                    prefix_counter.update({segs[0]: 1})
                    remain = ''.join(segs[1:])
                    remain_prefix_dict.setdefault(remain, set())
                    remain_prefix_dict[remain].add((k, segs[0]))
        return remain_prefix_dict

    def _get_candidates(self, word, pos, n=5):
        r"""
        Returns a list containing all possible words.

        :param word: str, the word to replace
        :param pos: str, the pos of the word to replace
        :param n: the number of returned words
        :return: a candidates list
        """
        assert pos is not None, "POS tag must be given!"
        candidates = []
        segs, _ = self.morpheme_analyzer.viterbi_segment(word)
        remain = ''.join(segs[1:])

        if remain in self.remain_prefix_dict:
            for type, prefix in self.remain_prefix_dict[remain]:
                if type == pos and prefix != segs[0]:
                    candidates.append(prefix + remain)
        return self.sample_num(candidates, n)

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
            if pos[index] in ['NN', 'JJ', 'RB', 'VB']:
                results.append(index)
        return results


if __name__ == "__main__":
    x = ['It', 'is', 'a', 'prefixed', 'string']
    y = ['DT', 'VBZ', 'DT', 'JJ', 'NN']

    data_sample = POSSample({'x': x, 'y': y})
    swap_ins = SwapPrefix()

    x = swap_ins.transform(sample=data_sample, field='x', n=3)

    for sample in x:
        print(sample.get_text('x'))

    """
    supposed output:
    It is a unfixed holding
    It is a affixed forwarding
    It is a transfixed holding
    """