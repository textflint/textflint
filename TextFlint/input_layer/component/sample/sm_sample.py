r"""
SM Sample Class
============================================

"""

from .sample import Sample
from ..field import Field
from ..field import TextField

__all__ = ['SMSample']


class SMSample(Sample):

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
        return 'SMSample'

    def check_data(self, data):
        r"""
        Check whether 'sentence1', 'sentence2 and 'y' is legal

        :param dict data: contains 'sentence1', 'sentence2', 'y' keys.
        :return:

        """
        assert 'sentence1' in data and isinstance(data['sentence1'], str), \
            "sentence1 should be in data, and the type of context should be str"
        assert 'sentence2' in data and isinstance(data['sentence2'], str), \
            "sentence2 should be in data, and the type of context should be str"
        assert 'y' in data, \
            "y should be in data, and y can be '0' or '1'"

    def load(self, data):
        """
        Convert data dict which contains essential information to SMSample.

        :param dict data: contains 'sentence1', 'sentence2', 'y' keys.
        :return:

        """
        self.sentence1 = TextField(data['sentence1'])
        self.sentence2 = TextField(data['sentence2'])
        self.y = Field(data['y'], field_type=str)

        if not self.is_legal():
            raise ValueError("y can only be '0' or '1'")

    def dump(self):
        return {'sentence1': self.sentence1.text,
                'sentence2': self.sentence2.text,
                'y': self.y.field_value,
                'sample_id': self.sample_id}

    def is_legal(self):
        r"""
        Validate whether the sample is legal

        :return: bool

        """
        if self.y.field_value not in ['0', '1']:
            return False
        return True
