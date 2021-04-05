"""
Text Field Class
=====================

A helper class that represents input string that to be modified.
"""

from .field import Field
from ....common.utils.list_op import *
from ....common.preprocess.en_processor import EnProcessor
from ....common.settings import ORIGIN, TASK_MASK, MODIFIED_MASK


class TextField(Field):
    """A helper class that represents input string that to be modified.

    Text that Sample contains parsed in data set,
    ``TextField`` provides multiple methods for Sample to modify.

    Support sentence level and word level modification, default using word level API.

    Attributes:
        field_value: str / list
            Sentence string or tokenized words.
        is_one_sent: bool
            whether input is a sentence
        split_by_space: bool
            splitting sentence by white or tokenizer

    """
    text_processor = EnProcessor()

    def __init__(
            self,
            field_value,
            mask=None,
            is_one_sent=False,
            split_by_space=False,
            **kwargs
    ):
        if isinstance(field_value, str):
            # lazy load
            self._words = None
            self._text = field_value
        elif isinstance(field_value, list):
            self._words = field_value
            self._text = None
        else:
            raise ValueError(
                'TextField supports string/token list, given {0}'.format(type(field_value)))

        super().__init__(field_value, field_type=(str, list), **kwargs)

        self._mask = None
        self.is_one_sent = is_one_sent
        self.split_by_space = split_by_space
        if mask:
            self.replace_mask(mask)

        # Process tags lazily
        self._sentences = None
        self._pos_tags = None
        self._ner_tags = None
        self._dp_tags = None

    def __hash__(self):
        return hash(self.text)

    def __len__(self):
        return len(self.words)

    @property
    def mask(self):
        if not self._mask:
            self._mask = [ORIGIN] * len(self.words)

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
            raise ValueError('Support mask value in {0}, ' 'while input mask value is {1}!'.format(
                [ORIGIN, TASK_MASK, MODIFIED_MASK], value))
        self._mask[index] = value

    def replace_mask(self, values):
        if not isinstance(values, list):
            raise ValueError(f"Cant replace mask values with {values}")
        if len(values) != len(self.words):
            raise ValueError(f"Mask values length {len(values)} unequal with words length {len(self.words)}")

        for index, value in enumerate(values):
            self.set_mask(index, value)

    def pos_of_word_index(self, desired_word_idx):
        """ Get pos tag of given index.

        Args:
            desired_word_idx: int

        Returns:
            pos tag of word of desired_word_idx.

        """
        if (desired_word_idx < 0) or (
                desired_word_idx > len(self.field_value)):
            raise ValueError(
                f"Cannot get POS tagging at index {desired_word_idx}")

        return self.pos_tagging[desired_word_idx]

    @staticmethod
    def _get_mirror_mask(mirror_list):
        """ Get list with all values MODIFIED_MASK and the same shape of mirror_list

        Args:
            mirror_list: list of list
                shape [[rep_0_0, ..., rep_0_i], ... , [rep_n_0, ..., rep_n_m]]

        Returns:

        """
        assert isinstance(mirror_list, list)
        mask_list = []

        for _list in mirror_list:
            mask_len = len(_list) if isinstance(_list, list) else 1
            mask_list.append([MODIFIED_MASK] * mask_len)

        return mask_list

    def replace_at_indices(self, indices, new_items):
        """ Replace words at indices and set their mask to MODIFIED_MASK.

        Args:
            indices: list of int/list/slice
                each index can be int indicate replace single item or their list like [1, 2, 3].
                each index can be list like (0,3) indicate replace items from 0 to 3(not included)
                    or their list like [(0, 3), (5,6)]
                each index can be slice which would be convert to list.
            new_items: list of str/list/tuple
                items corresponding indices.

        Returns:
            Replaced TextField object.
        """
        replace_mask = self._get_mirror_mask(new_items)
        mask_value = replace_at_scopes(self.mask, indices, replace_mask)
        field_value = replace_at_scopes(self.words, indices, new_items)

        return self.new_field(field_value, mask=mask_value)

    def replace_at_index(self, index, new_items):
        """ Replace words at indices and set their mask to MODIFIED_MASK.

        Args:
            index: int/list/slice
                can be int indicate replace single item or their list like [1, 2, 3].
                can be list like (0,3) indicate replace items from 0 to 3(not included)
                    or their list like [(0, 3), (5,6)]
                can be slice which would be convert to list.
            new_items: list of str/list/tuple
                items corresponding index.

        Returns:
            Replaced TextField object.
        """
        return self.replace_at_indices([index], [new_items])

    def delete_at_indices(self, indices):
        """ Delete words at indices and remove their mask value.

        Args:
            indices: list of int/list/slice
                each index can be int indicate replace single item or their list like [1, 2, 3].
                each index can be list like (0,3) indicate replace items from 0 to 3(not included)
                    or their list like [(0, 3), (5,6)]
                each index can be slice which would be convert to list.

        Returns:
            Modified TextField object.

        """
        mask_value = delete_at_scopes(self.mask, indices)
        field_value = delete_at_scopes(self.words, indices)

        return self.new_field(field_value, mask=mask_value)

    def delete_at_index(self, index):
        """ Delete words at index and remove their mask value.

        Args:
            index: int/list/slice
                can be int indicate replace single item or their list like [1, 2, 3].
                can be list like (0,3) indicate replace items from 0 to 3(not included)
                    or their list like [(0, 3), (5,6)]
                can be slice which would be convert to list.

        Returns:
            Modified TextField object.
        """

        return self.delete_at_indices([index])

    def insert_before_indices(self, indices, new_items):
        """ Insert words before indices.

        Args:
            indices: list of int
            new_items: list of str/list/tuple

        Returns:
            new field object.

        """
        insert_mask = self._get_mirror_mask(new_items)
        mask_value = insert_before_indices(self.mask, indices, insert_mask)
        field_value = insert_before_indices(self.words, indices, new_items)

        return self.new_field(field_value, mask=mask_value)

    def insert_before_index(self, index, new_items):
        """ Insert words before index and remove their mask value.

        Args:
             index: int
            new_items: str/list/tuple

        Returns:
            Modified TextField object.
        """
        return self.insert_before_indices([index], [new_items])

    def insert_after_indices(self, indices, new_items):
        """ Insert words after indices.

        Args:
            indices: list of int
            new_items: list of str/list/tuple

        Returns:
            Modified TextField object.

        """
        insert_mask = self._get_mirror_mask(new_items)
        mask_value = insert_after_indices(self.mask, indices, insert_mask)
        field_value = insert_after_indices(self.words, indices, new_items)

        return self.new_field(field_value, mask=mask_value)

    def insert_after_index(self, index, new_items):
        """ Insert words after index and remove their mask value.

        Args:
             index: int
            new_items: str/list/tuple

        Returns:
            Modified TextField object.
        """
        return self.insert_after_indices([index], [new_items])

    def swap_at_index(self, first_index, second_index):
        """ Swap items between first_index and second_index of origin_list

        Args:
            first_index: int
            second_index: int

        Returns:
             Modified TextField object.

        """
        mask_value = replace_at_scopes(
            self.mask, [first_index, second_index], [MODIFIED_MASK] * 2)
        field_value = swap_at_index(self.words, first_index, second_index)

        return self.new_field(field_value, mask=mask_value)

    @staticmethod
    def get_word_case(word):
        if len(word) == 0:
            return 'empty'

        if len(word) == 1 and word.isupper():
            return 'capitalize'

        if word.isupper():
            return 'upper'
        elif word.islower():
            return 'lower'
        else:
            for i, c in enumerate(word):
                if i == 0:  # do not check first character
                    continue
                if c.isupper():
                    return 'mixed'

            if word[0].isupper():
                return 'capitalize'
            return 'unknown'

    @property
    def words(self):
        if self._words or self._words == []:
            return self._words
        else:
            self._words = self.text_processor.word_tokenize(
                self.text, is_one_sent=self.is_one_sent, split_by_space=self.split_by_space)
            return self._words

    @property
    def sentences(self):
        if not self._sentences:
            self._sentences = self.text_processor.sentence_tokenize(self.text)

        return self._sentences

    @property
    def text(self):
        if self._text or self._text == '':
            return self._text
        else:
            if self.split_by_space:
                self._text = " ".join(self.words)
            else:
                self._text = self.text_processor.inverse_tokenize(self.words)
            return self._text

    @property
    def pos_tagging(self):
        """ Get NER tags.

        Example:
            given sentence 'All things in their being are good for something.'

            >> [('All', 'DT'),
                ('things', 'NNS'),
                ('in', 'IN'),
                ('their', 'PRP$'),
                ('being', 'VBG'),
                ('are', 'VBP'),
                ('good', 'JJ'),
                ('for', 'IN'),
                ('something', 'NN'),
                ('.', '.')]

        Returns:
            Tokenized tokens with their POS tags.

        """
        if not self._pos_tags:
            pos_tags = [pos for w, pos in self.text_processor.get_pos(self.words)]
            if len(pos_tags) != len(self.words):
                raise ValueError(
                    f"POS tagging not aligned with tokenized words")
            self._pos_tags = pos_tags

        return self._pos_tags

    @property
    def ner(self):
        """ Get NER tags.

        Example:
            given sentence 'Lionel Messi is a football player from Argentina.'

            >>[('Lionel Messi', 0, 2, 'PERSON'),
               ('Argentina', 7, 8, 'LOCATION')]

        Returns:
            A list of tuples, *(entity, start, end, label)*

        """
        if not self._ner_tags:
            self._ner_tags = self.text_processor.get_ner(
                self.words, return_char_idx=False)

        return self._ner_tags

    @property
    def dependency_parsing(self):
        """ Dependency parsing.

        Example:
            given sentence: 'The quick brown fox jumps over the lazy dog.'

            >>
                The	DT	4	det
                quick	JJ	4	amod
                brown	JJ	4	amod
                fox	NN	5	nsubj
                jumps	VBZ	0	root
                over	IN	9	case
                the	DT	9	det
                lazy	JJ	9	amod
                dog	NN	5	obl

        """
        if not self._dp_tags:
            self._dp_tags = self.text_processor.get_dep_parser(
                self.field_value,
                split_by_space=self.split_by_space,
                is_one_sent=self.is_one_sent)

        return self._dp_tags


if __name__ == "__main__":
    x = TextField(
        "Fudan University natural language processing group. Shanghai yangpu area.")
    print(x.field_value, x.words, x.sentences)

    print(x.pos_tagging)
    # test ner
    print(x.ner)
    print(x.dependency_parsing)

    print('--------------------test operations-------------------------')
    # test insert
    print(x.field_value)
    insert_before = x.insert_before_index(0, ['test '])
    print(insert_before.field_value)
    insert_before = x.insert_before_indices(
        [0, 2, 4], ['Wang', 'Xiao', ['fdu', 'jiangwan', 'cross_2']])
    print(insert_before.field_value)

    insert_after = x.insert_after_index(len(x.words) - 1, [' haha', 'test'])
    print(insert_after.field_value)
    insert_after = x.insert_after_indices(
        [0, 2, 7], ['Wang', 'Xiao', ['fdu', 'jiangwan', 'cross_2']])
    print(insert_after.field_value)

    # test swap
    swap = x.swap_at_index(0, 1)
    print(x.field_value)
    print(swap.field_value)

    # test delete
    delete = x.delete_at_index(0)
    print(x.field_value)
    print(delete.field_value)
    delete = x.delete_at_indices([0, [2, 4], len(x.words) - 1])

    # test replace
    replace = x.replace_at_index(0, '$')
    print(x.field_value)
    print(replace.field_value)
    replace = x.replace_at_indices(
        [0, [2, 4], [7, 8]], ['$', ['wang', 'xiao'], 'fDu'])
