r"""
ParallelWordSubstitute Class
============================================
"""
__all__ = ["ParallelWordSubstitute"]
from abc import abstractmethod

from ..word_substitute import WordSubstitute
from ....common.utils.error import FlintError
from ....common.utils import logger
from ....common.utils.list_op import trade_off_sub_words


class ParallelWordSubstitute(WordSubstitute):
    r"""
    Parallelly Word replace transformation to implement normal word replace functions.

    """

    def __init__(
        self,
        trans_min=1,
        trans_max=10,
        trans_p=0.1,
        stop_words=None,
        **kwargs
    ):
        r"""
        :param int trans_min: Minimum number of word will be augmented.
        :param int trans_max: Maximum number of word will be augmented. If None
            is passed, number of augmentation is
            calculated via aup_char_p. If calculated result from aug_p is
            smaller than aug_max, will use calculated
            result from aup_char_p. Otherwise, using aug_max.
        :param float trans_p: Percentage of word will be augmented.
        :param list stop_words: List of words which will be skipped from
            augment operation.
        :param ~textflint.common.preprocess.EnProcessor processor:
        :param bool get_pos: whether pass pos tag to _get_substitute_words API.

        """
        super().__init__(trans_min, trans_max, trans_p, stop_words)

    def __repr__(self):
        return 'ParallelWordSubstitute'

    def _transform(self, sample, field=['source', 'target'], n=1, **kwargs):
        r"""
        Transform text string according field.

        :param dict sample: input data, normally one data component.
        :param str fields: indicate which field to apply transformation
        :param int n: number of generated samples
        :return list: transformed sample list.

        """

        field.sort()
        source_tokens = sample.get_words(field[0])
        target_tokens = sample.get_words(field[1])
        source_tokens_mask = sample.get_mask(field[0])
        target_tokens_mask = sample.get_mask(field[1])

        # return up to (len(sub_indices) * n) candidates
        source_pos_info = sample.get_pos(field[0]) if self.get_pos else None
        target_pos_info = sample.get_pos(field[1]) if self.get_pos else None
        source_legal_indices = self.skip_aug(source_tokens, source_tokens_mask, pos=source_pos_info)
        target_legal_indices = self.skip_aug(target_tokens, target_tokens_mask, pos=target_pos_info)

        if not (source_legal_indices and target_legal_indices):
            return []

        sub_words, sub_indices = self._get_substitute_words(
            (source_tokens, target_tokens), (source_legal_indices, target_legal_indices),
            pos=(source_pos_info, target_pos_info), n=n)

        # select property candidates
        trans_num = self.get_trans_cnt(len(source_tokens))
        sub_words, sub_indices = trade_off_sub_words(
            sub_words, sub_indices, trans_num, n)

        if not sub_words:
            return []

        trans_samples = []
        for i in range(len(sub_words)):
            single_sub_words = sub_words[i]
            trans_samples.append(sample.replace_field_at_indices(field[0], [e[0] for e in sub_indices], single_sub_words).\
                replace_field_at_indices(field[1], [e[1] for e in sub_indices], single_sub_words))
        return trans_samples

    def transform(self, sample, n=1, field=['source', 'target'], split_by_space=False, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~textflint.input.component.sample.Sample sample: Data
            sample for augmentation.
        :param int n: Max number of unique augmented output, default is 5.
        :param str|list field: Indicate which fields to apply transformations.
        :param dict **kwargs: other auxiliary params.
        :return: list of Sample

        """
        if n < 1:
            return []

        if not isinstance(field, list):
            assert isinstance(field, str), "The type of field must be a str " \
                                           "or list not {0}".format(type(field))
            fields = [field]
        else:
            fields = field

        assert isinstance(fields, list), \
            "The type of field can choice in str or" \
            " list,not {0}".format(type(field))
        fields = list(set(fields))

        try:  # Deal with textflint Exception
            if len(fields) != 2:
                raise ValueError("The length of fields must be 2, not {0}.".format(len(fields)))
            else:
                if not ((fields[0] == 'source' and fields[1] == 'target') or 
                        (fields[0] == 'target' and fields[1] == 'source')):
                    raise ValueError("The elements of fields must be 'source' and 'target', not {0}.".format(fields))
                transform_results = self._transform(sample, n=n,
                                                    field=fields, split_by_space=split_by_space, **kwargs)
        except FlintError as e:
            logger.error(str(e))
            return []
        except Exception as e:
            logger.error(str(e))
            raise FlintError("You hit an internal error. "
                             "Please open an issue in "
                             "https://github.com/textflint/textflint"
                             " to report it.")
        if transform_results:
            return [sample for sample in transform_results
                    if (not sample.is_origin) and sample.is_legal()]
        else:
            return []

    def _get_substitute_words(self, words, legal_indices, pos=None, n=5):
        r"""
        Returns a list containing all possible words .

        :param list words: all words
        :param list legal_indices: indices which has not been skipped
        :param None|list pos: None or list of pos tags
        :param int n: max candidates for each word to be substituted
        :return list: list of list

        """
        source_legal_words = [words[0][index] for index in legal_indices[0]]
        target_legal_words = [words[1][index] for index in legal_indices[1]]
        source_legal_words_pos = [pos[0][index]
                           for index in legal_indices[0]] if self.get_pos else None
        target_legal_words_pos = [pos[1][index]
                           for index in legal_indices[1]] if self.get_pos else None

        candidates_list = []
        candidates_indices = []

        last_target_index = 0
        for index, word in enumerate(source_legal_words):
            _source_pos = source_legal_words_pos[index] if self.get_pos else None
            _target_pos = target_legal_words_pos[index] if self.get_pos else None
            candidates, target_index = self._get_candidates(word, (target_legal_words, legal_indices[1], last_target_index),
                    pos=_source_pos, n=n)
            # filter no word without candidates
            if candidates and target_index:
                candidates_indices.append([legal_indices[0][index], target_index])
                candidates_list.append(candidates)
                last_target_index = target_index + 1
        return candidates_list, candidates_indices

    @abstractmethod
    def _get_candidates(self, word, target, pos=None, n=5, **kwargs):
        r"""
        Returns a list containing all possible words .

        :param str word:
        :param str pos: the pos tag
        :param tuple target: target words and target indices
        :return list: candidates list

        """
        raise NotImplementedError