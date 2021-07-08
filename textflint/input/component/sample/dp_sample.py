r"""
DP Sample Class
============================================

"""

from .sample import Sample
from ..field import ListField, TextField
from ....common.utils.list_op import get_align_seq, normalize_scope

__all__ = ["DPSample"]


class DPSample(Sample):
    r"""
    DP Sample class to hold the data info and provide atomic operations.

    """

    def __repr__(self):
        return 'DPSample'

    def is_legal(self):
        r"""
        Validate whether the sample is legal

        """

        if not len(self.x.words) == len(self.postag.field_value) \
               == len(self.head.field_value) == len(self.deprel.field_value):
            return False
        for attr in (self.x, self.postag, self.head, self.deprel):
            if '' in attr.field_value:
                return False
        return True

    def check_data(self, data):
        assert 'word' in data and isinstance(data['word'], list), \
            "Word should be in data, and the type of word should be list"
        assert 'postag' in data and isinstance(data['postag'], list), \
            "Postag should be in data, and the type of postag should be list"
        assert 'head' in data and isinstance(data['head'], list), \
            "Head should be in data, and the type of head should be list"
        assert 'deprel' in data and isinstance(data['deprel'], list), \
            "Deprel should be in data, and the type of deprel should be list"

    def load(self, data):
        r"""
        Convert data dict to DPSample and get matched brackets.

        :param dict data: contains 'word', 'postag', 'head', 'deprel' keys.

        """
        assert data['deprel'][-1] == 'punct', \
            "The sentence should end with a punctuation."
        words = data['word']
        self.x = TextField(words)
        self.postag = ListField(data['postag'])
        self.head = ListField(data['head'])
        self.deprel = ListField(data['deprel'])

        try:
            left_bras = []
            match_bracket = []
            for i, word in enumerate(words):
                if word == '-LRB-':
                    left_bras.append(i + 1)
                if word == '-RRB-':
                    match_bracket.append((left_bras[-1], i + 1))
                    left_bras.pop(-1)
        except IndexError:
            raise TypeError('Missing matched brackets.')
        else:
            if left_bras:
                raise TypeError('Missing matched brackets.')
            else:
                self.brackets = match_bracket

        if not self.is_legal():
            raise ValueError("The postag data is not aligned with words.")

    def dump(self):
        if not self.is_legal():
            raise ValueError("The postag data is not aligned with words.")
        return {'word': self.x.words,
                'postag': self.postag.field_value,
                'head': self.head.field_value,
                'deprel': self.deprel.field_value,
                'sample_id': self.sample_id
                }

    def insert_field_after_indices(self, field, indices, items):
        r"""
        Insert items of multi given scopes before indices of field value
        at the same time.

        :param str field: Only value 'x' supported.
        :param list indices: shape：indices_num
        :param list items: shape: indices_num, correspond to indices
        :return ~DPSample: The sentence with words added.

        """
        assert isinstance(indices, list)
        assert isinstance(items, list)
        assert len(indices) == len(items)
        sample = self

        for index, new_item in enumerate(items):
            sample = sample.insert_field_after_index(
                field, indices[index], new_item)

        return sample

    def insert_field_after_index(self, field, ins_index, new_item):
        r"""
        Insert given data after the given index.

        :param str field: Only value 'x' supported.
        :param int ins_index: The index where the word will be inserted after.
        :param str new_item: The word to be inserted.
        :return ~DPSample: The sentence with one word added.

        """
        assert field == 'x'

        sample = self.clone(self)
        sample = super(
            DPSample,
            sample).insert_field_after_indices(
            field,
            [ins_index],
            [new_item])

        fields_to_values = {'postag': 'UNK', 'head': '0', 'deprel': 'unk'}

        for _field, value in fields_to_values.items():
            field_obj = getattr(sample, _field)
            insert_values = get_align_seq([new_item], value)[0]
            rep_obj = field_obj.insert_after_index(ins_index, insert_values)
            setattr(sample, _field, rep_obj)

        head_obj = sample.get_value('head')
        rep_obj = sample.head
        bias_len = len(new_item) if isinstance(new_item, list) else 1

        for i, head in enumerate(head_obj):
            head_id = int(head)
            if head_id > ins_index + 1:
                rep_obj = rep_obj.replace_at_index(i, str(head_id + bias_len))
            setattr(sample, 'head', rep_obj)

        return sample

    def insert_field_before_indices(self, field, indices, items):
        r"""
        Insert items of multi given scopes before indices of field value
        at the same time.

        :param str field: Only value 'x' supported.
        :param list indices: shape：indices_num
        :param list items: shape: indices_num, correspond to indices
        :return ~DPSample: The sentence with words added.

        """
        assert isinstance(indices, list)
        assert isinstance(items, list)
        assert len(indices) == len(items)
        sample = self.clone(self)

        for index, new_item in enumerate(items):
            sample = sample.insert_field_before_index(
                field, indices[index], new_item)

        return sample

    def insert_field_before_index(self, field, ins_index, new_item):
        r"""
        Insert given data before the given position.

        :param str field: Only value 'x' supported.
        :param int ins_index: The index where the word will be inserted after.
        :param str new_item: The word to be inserted.
        :return ~DPSample: The sentence with one word added.

        """
        assert field == 'x'

        sample = self.clone(self)
        sample = super(
            DPSample,
            sample).insert_field_before_indices(
            field,
            [ins_index],
            [new_item])

        fields_to_values = {'postag': 'UNK', 'head': '0', 'deprel': 'unk'}

        for _field, value in fields_to_values.items():
            field_obj = getattr(sample, _field)
            insert_values = get_align_seq([new_item], value)[0]
            rep_obj = field_obj.insert_before_index(ins_index, insert_values)
            setattr(sample, _field, rep_obj)

        head_obj = sample.get_value('head')
        rep_obj = sample.head
        bias_len = len(new_item) if isinstance(new_item, list) else 1

        for i, head in enumerate(head_obj):
            head_id = int(head)
            if head_id > ins_index:
                rep_obj = rep_obj.replace_at_index(i, str(head_id + bias_len))
            setattr(sample, 'head', rep_obj)

        return sample

    def delete_field_at_indices(self, field, indices):
        r"""
        Delete items of given scopes of field value.

        :param str field: Only value 'x' supported.
        :param list indices:
            shape：indices_num
            each index can be int indicate replace single item or their list
                like [1, 2, 3],
            can be list like (0,3) indicate replace items
                from 0 to 3(not included),
            can be slice which would be convert to list.
        :return ~DPSample: The sentence with words deleted.

        """
        assert isinstance(indices, list)
        sample = self.clone(self)

        for index in indices:
            sample = sample.delete_field_at_index(field, index)

        return sample

    def delete_field_at_index(self, field, del_index):
        r"""
        Delete data at the given position.

        :param str field: Only value 'x' supported.
        :param int|list|slice del_index:
            can be int indicate replace single item or their list
                like [1, 2, 3],
            can be list like (0,3) indicate replace items
                from 0 to 3(not included),
            can be slice which would be convert to list.
        :return ~DPSample: The sentence with one word deleted.

        """
        assert field == 'x'
        del_scope = normalize_scope(del_index)
        del_len = del_scope[1] - del_scope[0]

        sample = self.clone(self)
        sample = super(DPSample, sample)\
            .delete_field_at_indices(field, [del_scope])
        sample = super(DPSample, sample)\
            .delete_field_at_indices('postag', [del_scope])
        sample = super(DPSample, sample)\
            .delete_field_at_indices('head', [del_scope])

        sample = super(DPSample, sample)\
            .delete_field_at_indices('deprel', [del_scope])

        head_obj = sample.get_value('head')
        rep_obj = sample.head
        for i, head in enumerate(head_obj):
            head_id = int(head)
            # keep words dependency pointer which in del_scope may cause error
            # assume won't delete item with dependency pointers
            if head_id >= del_scope[1]:
                rep_obj = rep_obj.replace_at_index(i, str(head_id - del_len))
        setattr(sample, 'head', rep_obj)

        return sample


