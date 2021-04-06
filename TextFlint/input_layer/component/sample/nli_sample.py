r"""
NLI Sample Class
============================================

"""

from .sample import Sample
from ..field import Field, TextField

__all__ = ['NLISample']


class NLISample(Sample):
    def __init__(
        self,
        data,
        origin=None,
        sample_id=None
    ):
        r"""
        NLI Sample class to hold the necessary info
        and provide atomic operations.

        :param json data:  THe json obj that contains data info.
        :param ~Sample origin: Original sample obj.
        :param int sample_id: sample index

        """
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'NLISample'

    def check_data(self, data):
        r"""
        Check whether 'hypothesis', 'premise and 'y' is legal

        :param dict data: contains 'hypothesis', 'premise', 'y' keys.
        :return:

        """
        assert 'hypothesis' in data and isinstance(data['hypothesis'], str), \
            "hypothesis should be in data, and " \
            "the type of context should be str"
        assert 'premise' in data and isinstance(data['premise'], str), \
            "premise should be in data, and the type of context should be str"
        assert 'y' in data, \
            "y should be in data, and y can be " \
            "entailment, neutral and contradiction"

    def load(self, data):
        r"""
        Convert data dict which contains essential information to SMSample.

        :param dict data: contains 'hypothesis', 'premise', 'y' keys.
        :return:

        """
        self.hypothesis = TextField(data['hypothesis'])
        self.premise = TextField(data['premise'])
        self.y = Field(data['y'], field_type=str)

        if not self.is_legal():
            raise ValueError("y can only be entailment, neutral, "
                             "contradiction and non-entailment")

    def dump(self):
        return {'hypothesis': self.hypothesis.text,
                'premise': self.premise.text,
                'y': self.y.field_value,
                'sample_id': self.sample_id}

    def is_legal(self):
        r"""
        Validate whether the sample is legal

        :return: bool

        """
        if self.y.field_value not in ['entailment', 'neutral',
                                      'contradiction', 'non-entailment']:
            return False
        return True