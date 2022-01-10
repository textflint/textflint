r"""
DP CnSample Class
============================================

"""

from .cnsample import CnSample
from ..field import ListField, Field
from ....common.settings import ORIGIN, TASK_MASK, MODIFIED_MASK

__all__ = ['DPCnSample']


class DPCnTextField(Field):
    def __init__(self, field_value, mask=None):
        assert isinstance(field_value, list)
        self._words = field_value
        self._tokens = ''.join(field_value)

        if not mask:
            self._mask = [ORIGIN] * len(field_value)
        else:
            assert len(mask) == len(field_value)
            for index, mask_item in enumerate(mask):
                if mask_item not in [ORIGIN, TASK_MASK, MODIFIED_MASK]:
                    raise ValueError(
                        "Not support mask value of {0}".format(mask_item))
            self._mask = mask
        super().__init__(field_value, field_type=list)

    def __hash__(self):
        return hash(self._words)

    def __len__(self):
        return len(self._words)

    @property
    def words_length(self):
        return len(self._words)

    @property
    def mask(self):
        return self._mask[:]

    def set_mask(self, index, value):
        if not self._mask:
            self._mask = [ORIGIN] * len(self._words)

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
    def tokens(self):
        return self._tokens

    @property
    def words(self):
        return self._words


class DPCnSample(CnSample):

    def __init__(
        self,
        data,
        origin=None,
        sample_id=None
    ):
        r"""
        SM Sample class to hold the necessary info
        and provide atomic operations.

        :param json data:  THe json obj that contains data info.
        :param ~Sample origin: Original sample obj.
        :param int sample_id: sample index

        """
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'DPCnSample'

    def get_words(self, field):
        r"""
        Get tokenized words of given textfield

        :param str field: field name
        :return: tokenized words

        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, DPCnTextField), \
            f"{field} is not a dp text field, get words failed!"

        return field_obj.words[:]

    def get_tokens(self, field):
        r"""
        Get split sentences of given textfield

        :param str field: field name
        :return: list of sentences

        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, DPCnTextField), \
            f"{field} is not a dp text field, get token failed!"

        return field_obj.tokens[:]

    def get_mask(self, field):
        r"""
        Get word masks of given textfield

        :param str field: field name
        :return: list of mask values

        """
        field_obj = getattr(self, field)
        assert isinstance(field_obj, DPCnTextField), \
            f"{field} is not a dp text field, get mask failed!"

        return field_obj.mask[:]

    def normalize_pos(self, p):
        if p == 'NN' or p == 'NR':
            return 'n'
        elif p == 'VV':
            return 'v'
        else:
            return 'o'

    def get_pos(self, field):
        pos_info = []
        poses = self.pos.field_value
        words = self.x.words
        cnt = 0
        idx = 0
        for word in words:
            pos_info.append((self.normalize_pos(poses[cnt]), idx, idx+len(word)-1))
            cnt += 1
            idx += len(word)
        return pos_info


    def check_data(self, data):
        r"""
        Check whether 'sentence1', 'sentence2 and 'y' is legal

        :param dict data: contains 'sentence1', 'sentence2', 'y' keys.
        :return:

        """
        assert 'words' in data and isinstance(data['words'], list), \
            "words should be in data, and the type of context should be list"
        assert 'pos' in data and isinstance(data['pos'], list), \
            "pos should be in data, and the type of context should be list"
        assert 'heads' in data and isinstance(data['heads'], list), \
            "heads should be in data, and the type of context should be list"
        assert 'labels' in data and isinstance(data['labels'], list), \
            "labels should be in data, and the type of context should be list"

    def load(self, data):
        """
        Convert data dict which contains essential information to SMSample.

        :param dict data: contains 'sentence1', 'sentence2', 'y' keys.
        :return:

        """
        self.x = DPCnTextField(data['words'])
        self.y = ListField(data['labels'])
        self.pos = ListField(data['pos'])
        self.heads = ListField(data['heads'])

        self.col0 = ListField(data['0'])
        self.col2 = ListField(data['2'])
        self.col4 = ListField(data['4'])
        self.col5 = ListField(data['5'])
        self.col8 = ListField(data['8'])

    def dump(self):
        if not self.is_legal():
            raise ValueError("The postag data is not aligned with words.")
        return {
            '0': self.col0.field_value,
            'words': self.x.words,
            '2': self.col2.field_value,
            'pos': self.pos.field_value,
            '4': self.col4.field_value,
            '5': self.col5.field_value,
            'heads': self.heads.field_value,
            'labels': self.y.field_value,
            '8': self.col8.field_value,
        }

    def is_legal(self):
        r"""
        Validate whether the sample is legal

        :return: bool

        """
        if not len(self.x.words) == len(self.y.field_value) == len(self.pos.field_value) == len(self.heads.field_value):
            return False
        for attr in (self.x, self.y, self.pos, self.heads):
            if '' in attr.field_value:
                return False
        return True

    def unequal_replace_field_at_indices(self, field, indices, rep_items):
        r"""
        Replace scope items of field value with rep_items which may
        not equal with scope.

        :param field: field str
        :param indices: list of int/tupe/list
        :param rep_items: list
        :return: Modified Sample

        """
        assert len(indices) == len(rep_items) > 0
        # for i in range(len(rep_items)):
        #     rep_items[i] = [k for k in rep_items[i]]
        sample = self.clone(self)
        sorted_items, sorted_indices = zip(
            *sorted(zip(rep_items, indices), key=lambda x: x[1], reverse=True))
        for idx, sorted_item in enumerate(sorted_items):
            s, e = sorted_indices[idx]
            words = sample.get_words(field)
            mask = sample.get_mask(field)
            sorted_index = words.index(self.get_tokens(field)[s:e])
            words[sorted_index] = sorted_item
            mask[sorted_index] = MODIFIED_MASK
            setattr(sample, field, DPCnTextField(words, mask))
        return sample
