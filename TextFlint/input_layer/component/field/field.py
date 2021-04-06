"""
Field Base Class
=====================

"""


class Field:
    r"""
    A helper class that represents input string that to be modified.

    """

    def __init__(self, field_value, field_type=str, **kwargs):
        r"""
        :param string|int|list field_value: The string that Field represents.
        :param str field_type: field value type

        """
        self._field_type = field_type
        self.field_value = field_value

    def __eq__(self, other):
        """
        Compares two text instances to make sure they have the same text.

        """
        assert isinstance(other, Field)
        return self.field_value == other.field_value

    def __hash__(self):
        return hash(self.field_value)

    @property
    def field_value(self):
        return self._field_value

    @field_value.setter
    def field_value(self, value):
        # check field value type
        if self._field_type is not None \
                and not isinstance(value, self._field_type):
            raise ValueError("Invalid input type {0} (required {1})"
                             .format(type(value), self._field_type))

        self._field_value = value

    @property
    def field_type(self):
        return self._field_type

    @classmethod
    def new_field(cls, field_value, **kwargs):
        return cls(field_value, **kwargs)
