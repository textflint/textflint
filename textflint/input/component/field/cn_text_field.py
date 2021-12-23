"""
Text Field Class
=====================

A helper class that represents input string that to be modified.
"""

from .field import Field
from ....common.utils.list_op import *
from ....common.preprocess.cn_processor import CnProcessor
from ....common.settings import ORIGIN, TASK_MASK, MODIFIED_MASK

__all__ = ['CnTextField']


class CnTextField(Field):
    r"""
    A helper class that represents input string that to be modified.
    for example:
        1. CnTextField(['今天', '天气', '不错'],None)
        2. CnTextField('今天天气不错',None)

    :param str or list field_value: the value of the field.
    :param int mask: mask label.

    """

    cn_processor = CnProcessor()

    def __init__(self, field_value, mask=None):
        if isinstance(field_value, str):
            sentence = field_value
        elif isinstance(field_value, list):
            # join and re-tokenize because of insert/delete operation
            sentence = ''.join(field_value)
        else:
            raise ValueError(
                'TextField supports string/token list, given '
                '{0}'.format(type(field_value)))

        if not mask:
            self._mask = [ORIGIN] * len(sentence)
        else:
            assert len(mask) == len(sentence)
            for index, mask_item in enumerate(mask):
                if mask_item not in [ORIGIN, TASK_MASK, MODIFIED_MASK]:
                    raise ValueError(
                        "Not support mask value of {0}".format(mask_item))
            self._mask = mask

        super().__init__(sentence, field_type=str)
        self._text = sentence
        self._sentences = None
        self._tokens = [k for k in self._text]
        self._words = None
        self._ner_tags = None
        self._ner_list = None
        self._pos_tags = None
        self._dp = None

    def __hash__(self):
        return hash(self._text)

    def __len__(self):
        return len(self._text)

    @property
    def words_length(self):
        return len(self.words)

    @property
    def mask(self):
        return self._mask[:]

    def set_mask(self, index, value):
        if not self._mask:
            self._mask = [ORIGIN] * len(self.words)

        if index > len(self.mask) - 1:
            raise ValueError(
                "Index {0} out of range {1}".format(
                    index, len(
                        self.mask) - 1))
        if value not in [ORIGIN, TASK_MASK, MODIFIED_MASK]:
            raise ValueError(
                'Support mask value in {0},  while input mask value is {1}!'
                    .format([ORIGIN, TASK_MASK, MODIFIED_MASK], value)
            )
        self._mask[index] = value

    @property
    def text(self):
        return self._text

    @property
    def sentences(self):
        if not self._sentences:
            self._sentences = self.cn_processor.sentence_tokenize(self.text)
        return self._sentences

    @property
    def tokens(self):
        return self._tokens

    @property
    def words(self):
        if self._words or self._words == []:
            return self._words
        else:
            self._words = self.cn_processor.tokenize(
                self.text, cws=True
            )
            return self._words

    def ner(self):
        r"""
        ner function

        :return: ner tags, ner list
        """
        if not self._ner_tags:
            self._ner_tags, self._ner_list = self.cn_processor.get_ner(
                self.field_value)
            if len(self._ner_list) != len(self.field_value):
                raise ValueError(
                    f"Ner tagging not aligned with tokenized words")
        return self._ner_tags, self._ner_list

    def pos_tags(self):
        r"""
        pos tags fiction

        :return: pos tags
        """
        if not self._pos_tags:
            self._pos_tags = self.cn_processor.get_pos_tag(self.field_value)
        return self._pos_tags

    def dp(self):
        r'''
        dependency parsing function

        :return: dp
        '''
        if not self._dp:
            self._dp = self.cn_processor.get_dp(self.field_value)
        return self._dp

    def pos_of_word_index(self, desired_word_idx):
        r"""
        Get pos tag of given index.

        :param int desired_word_idx: desire index to get pos tag
        :return:  pos tag of word of desired_word_idx.

        """
        if (desired_word_idx < 0) or (
                desired_word_idx > len(self.field_value)):
            raise ValueError(
                f"Cannot get POS tagging at index {desired_word_idx}")

        return self._pos_tags[desired_word_idx][1]

    def insert_before_indices(self, indices, new_items):
        r"""
        Insert words before indices.
        :param [int] indices:
            can be int indicate replace single item or their list like [1, 2, 3]
            can be list like (0,3) indicate replace items
                from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list.
        :param [str|list|tuple] new_items: items corresponding index.
        :return: new TextField object.
        """
        for i in range(len(new_items)):
            if isinstance(new_items[i],str) and len(new_items[i]) > 1:
                new_items[i] = [k for k in new_items[i]]
            if isinstance(new_items[i],list):
                new_items[i] =  ''.join(new_items[i])
                new_items[i] = [k for k in new_items[i]]
        insert_mask = self._get_mirror_mask(new_items)
        mask_value = insert_before_indices(self.mask, indices, insert_mask)
        field_value = insert_before_indices(self.tokens, indices, new_items)

        return self.new_field(field_value, mask=mask_value)

    def insert_before_index(self, index, new_items):
        r"""
        Insert words before index and remove their mask value.
        :param int index:
            can be int indicate replace single item or their list like [1, 2, 3]
            can be list like (0,3) indicate replace items
                from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list.
        :param str|list|tuple new_items: items corresponding index.
        :return: new TextField object.
        """
        if isinstance(new_items,str) and len(new_items) > 1:
            new_items = [k for k in new_items]
        if isinstance(new_items,list):
            new_items =  ''.join(new_items)
            new_items = [k for k in new_items]
        return self.insert_before_indices([index], [new_items])

    def insert_after_indices(self, indices, new_items):
        r"""
        Insert words after indices.
        :param [int] indices:
            can be int indicate replace single item or their list like [1, 2, 3]
            can be list like (0,3) indicate replace items
                from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list.
        :param [str|list|tuple] new_items: items corresponding index.
        :return: new TextField object.
        """
        for i in range(len(new_items)):
            if isinstance(new_items[i],str) and len(new_items[i]) > 1:
                new_items[i] = [k for k in new_items[i]]
            if isinstance(new_items[i],list):
                new_items[i] =  ''.join(new_items[i])
                new_items[i] = [k for k in new_items[i]]
        insert_mask = self._get_mirror_mask(new_items)
        mask_value = insert_after_indices(self.mask, indices, insert_mask)
        field_value = insert_after_indices(self.tokens, indices, new_items)

        return self.new_field(field_value, mask=mask_value)

    def insert_after_index(self, index, new_items):
        r"""
        Insert words before index and remove their mask value.
        :param int index:
            can be int indicate replace single item or their list like [1, 2, 3]
            can be list like (0,3) indicate replace items
                from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list.
        :param str|list|tuple new_items: items corresponding index.
        :return: new TextField object.
        """
        if isinstance(new_items,str) and len(new_items) > 1:
            new_items = [k for k in new_items]
        if isinstance(new_items,list):
            new_items =  ''.join(new_items)
            new_items = [k for k in new_items]
        return self.insert_after_indices([index], [new_items])

    @staticmethod
    def _get_mirror_mask(mirror_list):
        r"""
        Get list with all values MODIFIED_MASK and the same shape of mirror_list

        :param list mirror_list:
            shape [[rep_0_0, ..., rep_0_i], ... , [rep_n_0, ..., rep_n_m]]
        :return: modified mask values with the same shape of mirror_list

        """
        assert isinstance(mirror_list, list)
        mask_list = []

        for _list in mirror_list:
            mask_len = len(_list) if isinstance(_list, list) else 1
            mask_list.append([MODIFIED_MASK] * mask_len)

        return mask_list

    def swap_at_index(self, first_index, second_index):
        r"""
        Swap items between first_index and second_index of origin_list

        :param int first_index: index of first item
        :param int second_index: index of second item
        :return:  Modified TextField object.

        """
        mask_value = replace_at_scopes(
            self.mask, [first_index, second_index], [MODIFIED_MASK] * 2)
        field_value = swap_at_index(self.tokens, first_index, second_index)

        return self.new_field(field_value, mask=mask_value)

    def delete_at_indices(self, indices):
        r"""
        Delete words at indices and remove their mask value.

        :param [int|list|slice] indices:
            each index can be int indicate replace single item
                or their list like [1, 2, 3].
            each index can be list like (0,3) indicate replace items
                from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            each index can be slice which would be convert to list.
        :return: Modified TextField object.

        """
        mask_value = delete_at_scopes(self.mask, indices)
        field_value = delete_at_scopes(self.tokens, indices)

        return self.new_field(field_value, mask=mask_value)

    def delete_at_index(self, index):
        r"""
        Delete words at index and remove their mask value.

        :param int|list|slice index:
            can be int indicate replace single item or their list like [1, 2, 3]
            can be list like (0,3) indicate replace items
                from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list.
        :return: Modified TextField object.

        """
        return self.delete_at_indices([index])

    def replace_at_indices(self, indices, new_items):
        r"""
        Replace words at indices and set their mask to MODIFIED_MASK.

        :param [int|list\slice] indices:
            each index can be int indicate replace single item
                or their list like [1, 2, 3].
            each index can be list like (0,3) indicate replace items
                from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            each index can be slice which would be convert to list.
        :param [str|list|tuple] new_items: items corresponding indices.
        :return: Replaced TextField object.

        """
        replace_mask = self._get_mirror_mask(new_items)
        mask_value = replace_at_scopes(self.mask, indices, replace_mask)
        field_value = replace_at_scopes(self.tokens, indices, new_items)

        return self.new_field(field_value, mask=mask_value)

    def replace_at_index(self, index, new_items):
        r"""
        Replace words at indices and set their mask to MODIFIED_MASK.

        :param int\list\slice index:
            can be int indicate replace single item or their list like [1, 2, 3]
            can be list like (0,3) indicate replace items
                from 0 to 3(not included) or their list like [(0, 3), (5,6)]
            can be slice which would be convert to list.
        :param str|list\tuple new_items: items corresponding index.
        :return: Replaced TextField object.

        """
        return self.replace_at_indices([index], [new_items])

