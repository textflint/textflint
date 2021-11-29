r"""
POS Sample Class
============================================

"""

from .sample import Sample
from ..field import ListField, TextField

__all__ = ['POSSample']


class POSSample(Sample):
    r"""
    POS Sample class to hold the necessary info and provide atomic operations.

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
        return 'POSSample'

    def get_pos(self, field):
        r"""
        Get text field pos tag.

        :param field: str
        :return: list, a pos tag list.

        """
        return self.y.field_value

    def check_data(self, data):
        r"""
        Check rare data format.

        """
        assert 'x' in data and isinstance(data['x'], list), \
            "x should be in data, and the type of x should be list"
        assert 'y' in data and isinstance(data['y'], list), \
            "y should be in data, and the type of y should be list"

    def is_legal(self):
        r"""
        Validate whether the sample is legal

        """
        if len(self.x.words) != len(self.y.field_value) \
                or len(self.x.words) == 0:
            return False
        return True

    def delete_field_at_indices(self, field, indices):
        r"""
        See sample.py for details.

        """
        sample = self.clone(self)
        field_obj = getattr(sample, field)

        assert isinstance(field_obj, (ListField, TextField))
        del_field = field_obj.delete_at_indices(indices)
        setattr(sample, field, del_field)

        if field == 'x':
            field_obj = getattr(sample, 'y')
            del_field = field_obj.delete_at_indices(indices)
            setattr(sample, 'y', del_field)
        else:
            raise Exception("Not support non-x fields!")

        return sample

    def insert_field_before_indices(self, field, indices, items):
        r"""
        See sample.py for details.

        """
        sample = self.clone(self)

        field_obj = getattr(sample, field)
        assert isinstance(field_obj, (ListField, TextField))
        rep_obj = field_obj.insert_before_indices(indices, items)
        setattr(sample, field, rep_obj)

        if field == 'x':
            field_obj = getattr(sample, 'y')
            if isinstance(items[0], str):
                rep_obj = field_obj.insert_before_indices(
                    indices, len(items) * ['UNK'])
            elif isinstance(items[0], list):
                rep_obj = field_obj.insert_before_indices(
                    indices, [['UNK'] * len(i) for i in items])
            else:
                raise Exception("Unsupported items types!")
            setattr(sample, 'y', rep_obj)
        else:
            raise Exception("Not support non-x fields!")

        return sample

    def insert_field_after_indices(self, field, indices, items):
        r"""
        See sample.py for details.

        """
        sample = self.clone(self)

        field_obj = getattr(sample, field)
        assert isinstance(field_obj, (ListField, TextField))
        rep_obj = field_obj.insert_after_indices(indices, items)
        setattr(sample, field, rep_obj)

        if field == 'x':
            field_obj = getattr(sample, 'y')
            if isinstance(items[0], str):
                rep_obj = field_obj.insert_after_indices(
                    indices, len(items) * ['UNK'])
            elif isinstance(items[0], list):
                rep_obj = field_obj.insert_after_indices(
                    indices, [['UNK'] * len(i) for i in items])
            else:
                raise Exception("Unsupported items types!")
            setattr(sample, 'y', rep_obj)
        else:
            raise Exception("Not support non-x fields!")

        return sample

    def unequal_replace_field_at_indices(self, field, indices, rep_items):
        r"""
        See sample.py for details.

        """
        assert len(indices) == len(rep_items) > 0
        sample = self.clone(self)
        sorted_items, sorted_indices = zip(
            *sorted(zip(rep_items, indices), key=lambda x: x[1], reverse=True))

        for idx, sorted_token in enumerate(sorted_items):
            sample = sample.delete_field_at_index(field, sorted_indices[idx])
            insert_index = sorted_indices[idx] if isinstance(
                sorted_indices[idx], int) else sorted_indices[idx][0]
            field_obj = getattr(sample, field)
            if insert_index > len(field_obj):
                raise ValueError(
                    'Cant replace items at range {0}'.format(
                        sorted_indices[idx]))
            elif insert_index == len(field_obj):
                sample = sample.insert_field_after_index(
                    field, insert_index - 1, sorted_token.split())
            else:
                sample = sample.insert_field_before_index(
                    field, insert_index, sorted_token.split())

        return sample

    def load(self, data):
        r"""
        Parse data into sample field value.

        """
        self.x = TextField(data['x'])
        self.y = ListField(data['y'])
        if not self.is_legal():
            raise ValueError("Data sample {0} is not legal, "
                             "pos tags mismatch words.".format(data))

    def dump(self):
        r"""
        Convert sample info to input data json format.

        """
        if not self.is_legal():
            raise ValueError("Data sample {0} is not legal, "
                             "pos tags mismatch words.".format(self))

        return {'x': self.x.words,
                'y': self.y.field_value,
                'sample_id': self.sample_id}
