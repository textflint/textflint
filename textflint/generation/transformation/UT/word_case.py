r"""
Word case transformation class
==========================================================
"""

__all__ = ['WordCase']

import random

from ..transformation import Transformation


class WordCase(Transformation):
    r"""
    Transforms an input to upper and lower case or capitalize case.
    A sentence can only be transformed into one sentence of each case at most.

    """
    def __init__(
        self,
        case_type='upper',
        **kwargs
    ):
        r"""
        :param case_type: case type, only support
            ['upper', 'lower', 'title', 'random']
        """
        super().__init__()
        if case_type not in ['upper', 'lower', 'title', 'random']:
            raise ValueError(
                'Not support {0} type, plz ensure case_type in {1}' .format(
                    case_type, [
                        'upper', 'lower', 'title', 'random']))
        self.case_type = case_type

    def __repr__(self):
        return 'WordCase' + '_' + self.case_type

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform each sample case according field.

        :param ~Sample sample: input data, normally one data sample.
        :param str field: indicate which filed to transform
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """
        field_value = sample.get_value(field)

        if self.case_type == 'random':
            case_type = ['upper', 'lower', 'title'][random.randint(0, 2)]
        else:
            case_type = self.case_type

        if isinstance(field_value, list):
            transform_text = [
                self.case_trans(
                    token,
                    case_type) for token in field_value]
        elif isinstance(field_value, str):
            transform_text = self.case_trans(field_value, case_type)
        else:
            raise ValueError(
                'Cant apply WordCase transformation to field {0}'.format(field))

        return [sample.replace_field(field, transform_text)]

    @staticmethod
    def case_trans(text, case_type):
        if case_type == 'upper':
            transform_text = text.upper()
        elif case_type == 'lower':
            transform_text = text.lower()
        else:
            transform_text = text.title()

        return transform_text
