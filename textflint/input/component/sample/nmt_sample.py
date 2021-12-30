r"""
NMT Sample Class
============================================

"""

from .sample import Sample
from ..field import TextField

__all__ = ['NMTSample']


class NMTSample(Sample):
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
        self.source = None
        self.target = None
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'NMTSample'

    def is_legal(self):
        """
        Validate whether the sample is legal

        """
        if len(self.source.text) > 0 and len(self.target.text) > 0:
            return True
        else:
            return False

    def check_data(self, data):
        r"""
        Check whether 'source' and 'target' is legal

        :param dict data: contains 'source', 'target' keys.
        :return:

        """
        assert 'source' in data and isinstance(data['source'], str), \
            "source should be in data, and the type of context should be str"
        assert 'target' in data and isinstance(data['target'], str), \
            "target should be in data, and the type of context should be str"

    def load(self, data):
        r"""
        Convert data dict which contains essential information to NMTSample.

        :param dict data: contains 'source', 'target' keys.
        :return:

        """
        self.source = TextField(data['source'])
        self.target = TextField(data['target'])
        if not self.is_legal():
            raise ValueError("Data sample " + str(data) + " is not legal, their length of text is 0")

    def dump(self):
        if not self.is_legal():
            raise ValueError("Data sample " + str(self.data) + " is not legal, the sorce and target are same or their length of text is 0")
        return {
            'source': self.source.text,
            'target': self.target.text,
            'sample_id': self.sample_id}

    def set_sample_id(self, sample_id):
        self.sample_id = sample_id


