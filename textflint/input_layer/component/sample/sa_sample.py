r"""
SA Sample Class
============================================

"""

from .sample import Sample
from ..field import Field, TextField
from collections import OrderedDict

__all__ = ['SASample']


class SASample(Sample):
    def __init__(
        self,
        data,
        origin=None,
        sample_id=None
    ):
        r"""

        :param dict data:  The json obj that contains data info.
        :param ~Sample origin: Original sample obj.
        :param int sample_id: sample index

        """
        self.x = None
        self.y = None
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'SASample'

    def is_legal(self):
        """
        Validate whether the sample is legal

        """
        if len(self.x.mask) == len(self.x.words):
            return True
        else:
            return False

    def check_data(self, data):
        r"""
        Check whether 'x' and 'y' is eagal

        :param dict data: contains 'x', 'y' keys.
        :return:

        """
        assert 'x' in data and isinstance(data['x'], str), \
            "x should be in data, and the type of context should be str"
        assert 'y' in data, "y should be in data"

    def load(self, data):
        r"""
        Convert data dict which contains essential information to SASample.

        :param dict data: contains 'x', 'y' keys.
        :return:

        """
        self.x = TextField(data['x'])
        self.y = Field(data['y'])
        if not self.is_legal():
            raise ValueError("Data sample {0} is not legal, "
                             "the mask spans mismatch x text".format(data))

    def dump(self):
        if not self.is_legal():
            raise ValueError("Mask spans mismatch x text")
        return {
            'x': self.x.text,
            'y': self.y.field_value,
            'sample_id': self.sample_id}

    def to_tuple(self):
        return OrderedDict({'x': self.x.text}), self.y.field_value

    def concat_token(self, max_name_len):
        r"""
        Find all the n-tuple from tokens that may be a name
        and splice it into a string

        :param int max_name_len: The list is composed of dicts which
            the key string is n-tuples spliced into strings, and the key indices
             is the index of n-tuples in the original sentence
        :return list tup_list

        """
        assert isinstance(max_name_len, int), "max_name_len must be type int"
        assert max_name_len > 0, 'max_name_len must be upper than zero'
        tokens = self.get_words('x')
        tup_list = []
        for i in range(len(tokens)):
            current_str = tokens[i]
            tup_list.append({'string': current_str, 'indices': [i, i + 1]})
            for j in range(1, max_name_len):
                if i + j < len(tokens):
                    current_str += ' %s' % tokens[i + j]
                    tup_list.append({'string': current_str,
                                     'indices': [i, i + j + 1]})
                else:
                    break
        return tup_list

