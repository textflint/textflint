import re
from ...transformation import Transformation
from ....input.component.sample.re_sample import RESample

r"""
AgeSwap class for age-related transformation 
"""
__all__ = ["SwapAge"]


class SwapAge(Transformation):
    r"""
    Entity position swap with paraphrase(age related)

    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

    def __repr__(self):
        return 'SwapAge'

    def _transform(self, sample, n=1, field='x', **kwargs):
        r"""
        Swap entity position through paraphrasing

        :param RESample sample: sample input
        :param int n: number of generated samples (no more than one)
        :return list: transformed sample list
        """
        assert(isinstance(sample, RESample)), \
            f"the type of 'sample' should be RESample, got " \
            f"{type(sample)} instead"
        assert(isinstance(n, int)), f"the type of 'n' should be int, " \
                                    f"got {type(n)} instead"

        regex = re.compile(r', (\d+) ,')
        words, relation = sample.get_sent()
        if 'age' not in relation:
            return [sample]
        ss, se, os, oe = sample.get_en()
        subj_type, obj_type, _ = sample.get_type()
        subj_words = words[ss:se+1]
        obj_words = words[os:oe+1]
        deprel, head = sample.get_dp()
        new_text = None
        trans_sample = {}

        if 'age' in relation and 'PERSON' in subj_type:
            match_obj = regex.match(' '.join(words[se + 1:se + 4]))
            if match_obj:
                if deprel[ss] == 'nsubj':  # find verb
                    words[ss:se+1] = ['[HEAD]']*len(subj_words)
                    words[os:oe+1] = ['[TAIL]']*len(obj_words)
                    verb_id = head[ss]
                    for i in range(len(words) - 1, verb_id - 1, -1):
                        if head[i] == verb_id and deprel[i] == 'punct':
                            break
                    #age = match_obj.group(1)
                    new_text = words[:se + 1] + words[se + 4:i]
                    new_text.append(',')
                    new_text.extend('and he was [TAIL] year old '
                                    'at that time .'.split(' '))
                    new_text.extend(words[i:])

        if new_text:
            ss, os = new_text.index('[HEAD]'), new_text.index('[TAIL]')
            se, oe = ss+new_text.count('[HEAD]')-1, \
                     os+new_text.count('[TAIL]')-1
            new_text[ss:se+1] = subj_words
            new_text[os:oe+1] = obj_words
            trans_sample['x'] = new_text
        else:
            trans_sample['x'] = words

        trans_sample['subj'], trans_sample['obj'], trans_sample['y'] = \
            [ss, se], [os, oe], relation
        trans_samples = sample.replace_sample_fields(trans_sample)
        return [trans_samples]


