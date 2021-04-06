"""
CWS Sample Class
============================================
"""

from .sample import Sample
from ..field.cn_text_field import CnTextField
from ..field import ListField
from ....common.settings import ORIGIN, MODIFIED_MASK
from ....common.utils.list_op import *

__all__ = ['CWSSample']


class CWSSample(Sample):
    r"""
    Our segmentation rules are based on ctb6.

    the input x can be a list or a sentence
    the input y is segmentation label include:B,M,E,S
    the y also can automatic generation,if you want automatic generation
        you must input an empty list and x must each word in x is separated by
        a space or split into each element of the list
    Note that punctuation should be separated into a single word

    Example::

        1. input {'x':'小明好想送Jo圣诞礼物', 'y' = ['B', 'E', 'B', 'E', 'S', 'B',
            'E', 'B', 'E', 'B', 'E']}
        2. input {'x':['小明','好想送Jo圣诞礼物'], 'y' = ['B', 'E', 'B', 'E', 'S',
            'B', 'E', 'B', 'E', 'B', 'E']}
        3. input {'x':'小明 好想 送 Jo 圣诞 礼物', 'y' = []}
        4. input {'x':['小明', '好想', '送', 'Jo', '圣诞', '礼物'], 'y' = []}

    """

    def __init__(self, data, origin=None, sample_id=None):
        r"""
        :param dict data: The dict obj that contains data info
        :param int sample_id: the id of sample
        :param bool origin: if the sample is origin

        """
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'CWSSample'

    @staticmethod
    def is_legal():
        return True

    def check_data(self, data):
        r"""
        Check the whether the data legitimate but we don't check that the label
        is correct if the data is not legal but acceptable format, change
        the format of data

        :param dict data: The dict obj that contains data info

        """
        assert 'x' in data and 'y' in data
        assert isinstance(data['y'], list), \
            f"The type of data[y] me be list,not {0}".format(type(data['y']))
        assert isinstance(data['x'], (str, list)), \
            f"The type of data[y] me be list or str, not {0}"\
                .format(type(data['x']))

        if not data['y']:
            if isinstance(data['x'], str):
                data['x'] = data['x'].strip().split(' ')
            data['y'] = []
            sentence = ''
            for x in data['x']:
                x.replace(' ', '')
                sentence += x
                if len(x) == 1:
                    data['y'] += ['S']
                elif len(x) > 1:
                    data['y'] += ['B'] + ['M'] * (len(x) - 2) + ['E']
            data['x'] = sentence

        else:
            sentence = []
            for i in data['x']:
                sentence += i.replace(' ', '')
            data['x'] = sentence

        cws_tag = ['B', 'M', 'E', 'S']
        assert len(data['x']) == len(data['y'])
        for tag in data['y']:
            assert tag in cws_tag

    def load(self, data):
        r"""
        Convert data dict which contains essential information to CWSSample.

        :param dict data: The dict obj that contains data info
        """
        self.x = CnTextField(data['x'])
        assert isinstance(data['y'], list)
        self.y = ListField(data['y'])

    def dump(self):
        assert len(
            self.x.mask) == len(
            self.x.field_value) == len(
            self.y.field_value)
        return {
            'x': self.x.field_value,
            'y': self.y.field_value,
            'sample_id': self.sample_id}

    @property
    def mask(self):
        return self.x.mask

    @property
    def pos_tags(self):
        return self.x.pos_tags()

    @property
    def ner(self):
        return self.x.ner()

    def get_words(self):
        r"""
        Get the words from the sentence.

        :return list: the words in sentence

        """
        start = 0
        words = []
        while start < len(self.x.field_value):
            # find the word
            if self.y.field_value[start] == 'B':
                end = start + 1
                while self.y.field_value[end] != 'E':
                    end += 1
            elif self.y.field_value[start] == 'S':
                end = start
            else:
                raise ValueError(f"the label is not right")
            words.append(self.x.field_value[start:end + 1])
            start = end + 1
        return words

    def replace_at_ranges(self, indices, new_items, y_new_items=None):
        r"""
        Replace words at indices and set their mask to MODIFIED_MASK.

        :param list indices: The list of the pos need to be changed.
        :param list new_items: The list of the item need to be changed.
        :param list y_new_items: The list of the mask info need to be changed.
        :return: replaced CWSSample object.

        """
        indices, items, mask, y_new_items = self.check(
            indices, new_items, y_new_items)
        if len(indices):
            cws = self.clone(self)
            new_mask = unequal_replace_at_scopes(self.mask, indices, mask)
            new_field = unequal_replace_at_scopes(self.x.token, indices, items)
            x = self.x.new_field(new_field, mask=new_mask)
            setattr(cws, 'x', x)
            if y_new_items:
                y = unequal_replace_at_scopes(
                    self.y.field_value, indices, y_new_items)
                setattr(cws, 'y', ListField(y))
            return cws
        else:
            return self

    def update(self, x, y):
        r"""
        Replace words at indices and set their mask to MODIFIED_MASK.

        :param str x: the new sentence.
        :param list y: the new labels.
        :return: new CWSSample object.
        """
        cws = self.clone(self)
        setattr(cws, 'x', x)
        setattr(cws, 'y', ListField(y))
        return cws

    def check(self, indices, new_items, y_new_items=None):
        r"""
        Check whether the position of change is legal.

        :param list indices: The list of the pos need to be changed.
        :param list new_items: The list of the item need to be changed.
        :param list y_new_items: The list of the mask info need to be changed.
        :return three list: legal position, change items, change labels.
        """

        assert len(indices) == len(new_items)
        legal_indices = []
        legal_items = []
        mask_change = []
        legal_y = []
        mask = self.mask
        for i in range(len(indices)):
            flag = True
            if isinstance(indices[i], list):
                for j in range(indices[i][0], indices[i][1]):
                    if mask[j] != ORIGIN:
                        flag = False
                        break
            else:
                if mask[indices[i]] != ORIGIN:
                    flag = False
            if flag:
                legal_indices.append(indices[i])
                legal_items.append(new_items[i])
                if y_new_items:
                    legal_y.append(y_new_items[i])
                if isinstance(new_items[i], list):
                    change = []
                    for k in new_items[i]:
                        change += [MODIFIED_MASK] * len(k)
                    mask_change.append(change)
                else:
                    mask_change.append([MODIFIED_MASK] * len(new_items[i]))

        return legal_indices, legal_items, mask_change, legal_y

    @staticmethod
    def get_labels(words):
        r"""
        Get the label of the word.

        :param str words: The word you want to get labels.
        :return list: the label of the words.
        """
        assert isinstance(words, str), \
            "The type of words must be str, not {0}".format(type(words))
        if len(words) == 1:
            return ['S']
        return ['B'] + ['M'] * (len(words) - 2) + ['E']
