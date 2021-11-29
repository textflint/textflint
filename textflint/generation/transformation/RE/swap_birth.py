r"""
BirthSwap class for birth-related transformation
"""
from ....input.component.sample import RESample
from ...transformation import Transformation

__all__ = ["SwapBirth"]


class SwapBirth(Transformation):
    r"""
    Entity position swap with paraphrase(birth related)

    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

    def __repr__(self):
        return 'SwapBirth'

    def generate_birth_for_root(self, idx, tokens, pos_tags, heads):
        r"""generate new sentence if birth is root

        :param int idx: the idx of word "bear" in sentence
        :param list tokens: tokens of the sentence
        :param list pos_tags: pos tagging labels of the sentence
        :param list heads: stanford heads

        :return string: transformed sentence

        """
        new_sen = tokens[:idx - 1]

        v = -1
        for pos in range(idx + 1, len(tokens)):
            if pos_tags[pos].startswith('V') and heads[pos] == idx + 1:
                v = pos
        if v < 0:
            return []
        else:
            born = tokens[idx - 1:v]
            if tokens[v - 1] == 'and':
                born.pop()
            new_sen.extend([','] + tokens[v:-1] + [','] + born + ['.'])

        return new_sen

    def generate_birth_for_clause(self, idx, tokens, deprels):
        r"""generate new sentence if birth is clause

        :param int idx: the idx of word "bear" in sentence
        :param list tokens: tokens of the sentence
        :param list deprels: stanford dependency relations

        :return string: transformed sentence

        """
        root_id = -1
        for k, r in enumerate(deprels):
            if r == 'ROOT':
                root_id = k
        if root_id < idx:
            return []
        end = root_id - 1
        while tokens[end] != ',':
            end = end - 1
        born = tokens[idx:end]
        if idx - 1 >= 0 and tokens[idx - 1] == ',':
            new_sen = tokens[:idx - 1] + tokens[end + 1:]
        elif idx - 2 >= 0 and tokens[idx - 2] == 'who':
            if idx - 3 >= 0 and tokens[idx - 3] == ',':
                new_sen = tokens[:idx - 3] + tokens[end + 1:]
            else:
                new_sen = tokens[:idx - 2] + tokens[end + 1:]
        else:
            return []

        new_sen = ["Born"] + born[1:] + [","] + new_sen

        return new_sen

    def generate_new_sen_for_birth(self, idx, tokens, pos_tags, heads, deprels):
        r"""
        generate new sentence

        :param int idx: the idx of word "bear" in sentence
        :param list tokens: tokens of the sentence
        :param list pos_tags: pos tagging labels of the sentence
        :param list heads: stanford heads
        :param list deprels: stanford dependency relations
        :return string: transformed sentence
        """
        assert (isinstance(idx, int)), \
            f"the type of 'idx' should be int, got {type(idx)} instead"
        assert (isinstance(tokens, list)), \
            f"the type of 'tokens' should be list, got {type(tokens)} instead"
        assert (isinstance(pos_tags, list)), \
            f"the type of 'pos_tags' should be list, got " \
            f"{type(pos_tags)} instead"
        assert (isinstance(heads, list)), \
            f"the type of 'heads' should be list, got {type(heads)} instead"
        assert (isinstance(deprels, list)), \
            f"the type of 'deprels' should be list, got {type(deprels)} instead"
        assert (len(tokens) == len(pos_tags) == len(heads) == len(deprels)), \
            f"the length of the list inputs should be the same, got " \
            f"{len(tokens), len(pos_tags), len(heads), len(deprels)} instead"
        assert (idx >= 0 and idx < len(tokens)), \
            f"got invalid value of idx: {idx}"
        assert (isinstance(heads[0], int)), \
            f"the type of each token in 'heads' " \
            f"should be int, got {type(heads[0])} instead"
        assert (len([i for i in heads if i < 0 or i >= len(tokens)]) == 0), \
            f"got invalid value of 'heads': {heads}"
        assert ("ROOT" in deprels), f"'ROOT' should be " \
                                    f"included in 'deprels', got {deprels}"
        assert (0 in heads), f"0 should be included in 'heads', got {heads}"

        deprel = deprels[idx]

        ### if "born" is root of the sent, swap the position of born- and clause
        if deprel == 'ROOT':
            new_sen = self.generate_birth_for_root(idx, tokens, pos_tags, heads)

        else:
            new_sen = self.generate_birth_for_clause(idx, tokens, deprels)

        return new_sen

    def _transform(self, sample, n=1, field='x', **kwargs):
        r"""
        Swap entity position through paraphrasing

        :param RESample sample: sample input
        :param int n: number of generated samples (no more than one)
        :return list: transformed sample list 
        """
        assert (isinstance(sample, RESample)), \
            f"the type of 'sample' should be RESample, " \
            f"got {type(sample)} instead"
        assert (isinstance(n, int)), f"the type of 'n' should be int, " \
                                     f"got {type(n)} instead"

        words, relation = sample.get_sent()
        if 'birth' not in relation:
            return [sample]
        sh, st, oh, ot = sample.get_en()
        pos_tag = sample.get_pos(field)
        head_pos = [sh, st]
        subj = ' '.join(words[sh:st + 1])
        obj = ' '.join(words[oh:ot + 1])
        new_sample = {}
        new_sample['x'] = words

        if 'birth' in relation:
            deprels, heads = sample.get_dp()
            new_sen = []
            words[sh:st + 1] = ["<SUBJ>"] * (st - sh + 1)
            words[oh:ot + 1] = ["<OBJ>"] * (ot - oh + 1)
            for i, word in enumerate(words):
                if word == "born":
                    if head_pos[1] < i:
                        new_sen = self.generate_new_sen_for_birth(
                            i, words, pos_tag, heads, deprels)

            if new_sen:
                new_sh = new_sen.index("<SUBJ>")
                new_st = new_sh + (st - sh)
                new_oh = new_sen.index("<OBJ>")
                new_ot = new_oh + (ot - oh)
                sh, st, oh, ot = new_sh, new_st, new_oh, new_ot
                new_sen[new_sh:new_st + 1] = subj.split()
                new_sen[new_oh:new_ot + 1] = obj.split()
                new_sample['x'] = new_sen

        new_sample['subj'], new_sample['obj'], new_sample['y'] = \
            [sh, st], [oh, ot], relation
        trans_samples = sample.replace_sample_fields(new_sample)

        return [trans_samples]
