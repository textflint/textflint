"""
UT Sample Class
============================================

"""

from .sample import Sample
from ..field import TextField

__all__ = ['UTSample']


class UTSample(Sample):
    """ Universal Transformation sample.

    Universe Transformation is not a subtask of NLP,
    implemented for providing universal text transformation function.

    """

    def __init__(self, data, origin=None, sample_id=None):
        self.x = None
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'UTSample'

    def check_data(self, data):
        assert 'x' in data and isinstance(data['x'], str)

    def load(self, data):
        """ Convert data dict which contains essential information to SASample.

        Args:
            data: dict
                contains 'x' key at least.

        """
        self.x = TextField(data['x'])

    def dump(self):
        return {'x': self.x.text, 'sample_id': self.sample_id}


if __name__ == "__main__":
    sent = "this is a sentence aa bnn cc d"
    ut_sample = UTSample({'x': sent})
    new_sample = ut_sample.replace_field('x', "I cant believe this sentence!")
    print(new_sample.is_origin)
    print(new_sample.dump())
