"""
SAcn Sample
============================================

"""

from .cnsample import CnSample
from ..field import CnTextField

__all__ = ['SACnSample']


class SACnSample(CnSample):
    r"""
    Example:
        input {'x':'住起来很舒服', 'y' = ’1‘}

    """
    def __init__(
        self,
        data,
        origin=None,
        sample_id=None
    ):
        self.x = None
        self.y = None
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'SACnSample'

    def check_data(self, data):
        assert 'x' in data and isinstance(data['x'], str)
        assert 'y' in data, "y should be in data"
        assert data['y'] in ['0', '1'], "y should be \'0\' or \'1\'"

    def load(self, data):
        r"""
        Convert data dict which contains essential information to SASample.

        :param dict data: contains 'x' key at least.
        :return:

        """
        self.x = CnTextField(data['x'])
        self.y = CnTextField(data['y'])

    def dump(self):
        return {
            'x': self.x.field_value,
            'y': self.y.field_value,
            'sample_id': self.sample_id}

    def is_legal(self):
        r"""
        Validate whether the sample is legal

        :return: bool
        """
        return True
