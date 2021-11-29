r"""
EmployeeSwap class for employee-related transformation
"""

__all__ = ["SwapEmployee"]
from ...transformation import Transformation
from ....common.settings import TITLE
from ....common.utils.install import download_if_needed
from ....common.utils.load import json_loader
from ....input.component.sample.re_sample import RESample


class SwapEmployee(Transformation):
    r"""
    Entity position swap with paraphrase(employee related)

    """
    titles_dict = json_loader(download_if_needed(TITLE))

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

    def __repr__(self):
        return 'SwapEmployee'

    def split_sent(self, head_pos, tail_pos, words):
        r"""
        split sentence into 3 pieces: left, middle and right.

        :param list head_pos: position of subject entity
        :param list tail_pos: position of object entity
        :param list words: sentence tokens
        :return bool: whether to reverse position of subject entity and
            object entity
                list: entity placed on the left
                list: entity placed on the right
                list: token indices place between left entity and right entity
                list: token place between left entity and right entity
                list: tokens place on the left of left entity
                list: tokens place on the right of right entity
        """

        assert(isinstance(head_pos, list)), \
            f"the type of 'head_pos' " \
            f"should be list, got {type(head_pos)} instead"
        assert (isinstance(tail_pos, list)), \
            f"the type of 'tail_pos' should be list, " \
            f"got {type(tail_pos)} instead"
        assert(isinstance(words, list)), \
            f"the type of 'words' should be list, " \
            f"got {type(words)} instead"
        assert(len(head_pos) == 2 and len(tail_pos) == 2), \
            f"the length of pos should be 2, " \
            f"got input length of {len(head_pos)} or {len(tail_pos)} instead"
        assert (head_pos[0]<=head_pos[1] and tail_pos[0]<=tail_pos[1]), \
            f"start index of entity should not be greater " \
            f"than end index, got {head_pos[0]}>{head_pos[1]} " \
            f"or {tail_pos[0]}>{tail_pos[1]} instead"
        assert (head_pos[0] >= 0 and tail_pos[0] >= 0), \
            f"start index of entity should be greater than 0, got \
            {head_pos[0]}<0 or {tail_pos[0]}<0 instead"
        assert (head_pos[1] < len(words) and tail_pos[1] < len(words)), \
            f"end index of entity should not be greater " \
            f"than the length of words, got \
            {head_pos[1]}>={len(words)} or {tail_pos[1]}>={len(words)} instead"

        if head_pos[-1] < tail_pos[0]:
            pre = head_pos
            post = tail_pos
            reverse = False
        elif tail_pos[-1] < head_pos[0]:
            reverse = True
            pre = tail_pos
            post = head_pos
        else:
            return None

        left = words[pre[0]:pre[1] + 1]
        right = words[post[0]:post[1] + 1]
        middle_pos = [pre[-1] + 1, post[0]]
        middle_words = words[pre[-1] + 1:post[0]]
        left_words = words[:pre[0]]
        right_words = words[post[1] + 1:]

        return reverse, left, right, middle_pos, middle_words, \
               left_words, right_words

    def assert_attributive(self, left, right, words, heads,
                           middle_words, middle_pos):
        r"""
        Judge whether sentence piece between entities is attributive or not.

        :param list left: entity placed on the left
        :param list right: entity placed on the right
        :param list words: sentence tokens
        :param list middle_pos: token indices place between left
            entity and right entity
        :param list middle_words: token place between left entity
            and right entity
        :return bool : indicator or whether the middle part is
            attributive or not

        """
        assert(isinstance(left, list)), \
            f"the type of 'left' should be list, got {type(left)} instead"
        assert (isinstance(right, list)), \
            f"the type of 'right' should be list, got {type(right)} instead"
        assert (isinstance(words, list)), \
            f"the type of 'words' should be list, got {type(words)} instead"
        assert (isinstance(heads, list)), \
            f"the type of 'heads' should be list, got {type(heads)} instead"
        assert (isinstance(middle_pos, list)), \
            f"the type of 'middle_pos' should be list, got " \
            f"{type(middle_pos)} instead"
        assert (isinstance(middle_words, list)), \
            f"the type of 'middle_words' should be list, got " \
            f"{type(middle_words)} instead"
        assert (len(heads) == len(words)), \
            f"the length of 'heads' should be equal with " \
            f"the length of 'words', got {len(heads)} and {len(words)} instead"
        assert (len([i for i in heads if i < 0 or i >= len(words)]) == 0), \
            f"got invalid value of 'heads': {heads}"

        is_attrib = True
        ent_words = left + right
        for word, pos in zip(middle_words, range(*middle_pos)):
            if word == ',':
                continue
            if heads[pos] != 0 and words[heads[pos] - 1] not in ent_words and (
                    heads[pos] < middle_pos[0] + 1 or
                    heads[pos] > middle_pos[1]):
                is_attrib = False

        return is_attrib

    def generate_new_item(self, reverse, left, right, left_words,
                          right_words, middle_words, title_pos):
        r"""
        split sentence into 3 pieces: left, middle and right.

        :param bool reverse: if the position of head and tail entity is reversed
        :param list left: entity placed on the left
        :param list right: entity placed on the right
        :param list left_words: tokens place on the left of left entity
        :param list right_words: tokens place on the right of right entity
        :param list middle_words: token place between left entity
            and right entity
        :param list title_pos: the position of TITLE
        : return list: new list of words
                 list: the position of subject entity
                 list: the position of object entity

        """
        assert (isinstance(left_words, list)), \
            f"the type of 'left_words' should be list, " \
            f"got {type(left_words)} instead"
        assert (isinstance(right_words, list)), \
            f"the type of 'right_words' should be list, " \
            f"got {type(right_words)} instead"
        assert (isinstance(left, list)), \
            f"the type of 'left_words' should be list, " \
            f"got {type(left)} instead"
        assert (isinstance(right, list)), \
            f"the type of 'right_words' should be list, " \
            f"got {type(right)} instead"
        assert (isinstance(middle_words, list)), \
            f"the type of 'right_words' should be list, " \
            f"got {type(middle_words)} instead"
        assert (isinstance(title_pos, list)), \
            f"the type of 'title_pos' should be list, " \
            f"got {type(title_pos)} instead"
        assert (isinstance(reverse, bool)), \
            f"the type of 'reverse' should be bool, " \
            f"got {type(reverse)} instead"
        assert (title_pos[1] <= len(middle_words)), \
            f"the end of 'title_pos' should not be larger than " \
            f"the length of 'middle_words', " \
            f"got {title_pos[1]}>{len(middle_words)} instead"

        new_middle_words = list(middle_words[title_pos[1]:])
        new_words = left_words + left + new_middle_words + right + right_words
        left_pos = [len(left_words), len(left_words) + len(left) - 1]
        right_pos = [len(left_words) + len(left) + len(new_middle_words),
                     len(left_words) + len(left) + len(new_middle_words)
                     + len(right) - 1]

        if not reverse:
            sh, st = left_pos
            oh, ot = right_pos
        else:
            sh, st = right_pos
            oh, ot = left_pos

        return new_words, [sh, st], [oh, ot]

    def _transform(self, sample, n=5, **kwargs):
        r"""
        Swap entity position through paraphrasing

        :param RESample sample: sample input
        :param int n: number of generated samples (no more than one)
        :return list: transformed sample list

        """
        assert(isinstance(sample, RESample)), \
            f"the type of 'sample' should be RESample, " \
            f"got {type(sample)} instead"
        assert(isinstance(n, int)), f"the type of 'n' should be int, " \
                                    f"got {type(n)} instead"

        tokens, relation = sample.get_sent()
        if 'emplo' not in relation and "business" not in relation:
            return [sample]
        sh, st, oh, ot = sample.get_en()
        head_pos, tail_pos = [sh,st], [oh,ot]
        subj = [sh, st]
        obj = [oh, ot]
        _, heads = sample.get_dp()
        new_sample = {}
        new_sample['x'] = tokens

        if "emplo" or "business" in relation:
            ner = sample.stan_ner_transform()
            assert len(ner) == len(tokens)
            splited_sent = self.split_sent(head_pos, tail_pos, tokens)
            if splited_sent is not None:
                reverse, left, right, middle_pos, \
                middle_words, left_words, right_words = splited_sent

                is_attrib = self.assert_attributive(
                    left, right, tokens, heads, middle_words, middle_pos)

                middle_text = " ".join(middle_words)
                is_title = False
                title_pos = [0, 0]
                for title in self.titles_dict:
                    if title.lower() in middle_text.lower():
                        is_title = True
                        title_pos[0] = len((middle_text.split(
                            title.lower())[0]).split(" "))
                        title_pos[1] = title_pos[0] + len(title.split(" ")) - 1
                if is_title and is_attrib:
                    new_sample['x'], subj, obj = self.generate_new_item(
                        reverse, left, right, left_words, right_words,
                        middle_words, title_pos)

        new_sample['subj'], new_sample['obj'], new_sample['y'] \
            = subj, obj, relation
        trans_samples = sample.replace_sample_fields(new_sample)

        return [trans_samples]
