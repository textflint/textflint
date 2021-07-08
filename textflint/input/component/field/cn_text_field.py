"""
CnTextField Class
============================================
"""

from ....common.preprocess.cn_processor import CnProcessor
from .field import Field
from ....common.settings import ORIGIN, TASK_MASK, MODIFIED_MASK

__all__ = ['CnTextField']


class CnTextField(Field):
    r"""
    A helper class that represents input string that to be modified.

    :param str or list field_value: the value of the field.
    :param int mask: mask label.

    """

    cn_processor = CnProcessor()

    def __init__(self, field_value, mask=None):
        if isinstance(field_value, str):
            sentence = field_value
        elif isinstance(field_value, list):
            # join and re-tokenize because of insert/delete operation
            sentence = ''.join(field_value)
        else:
            raise ValueError(
                'TextField supports string/token list, given '
                '{0}'.format(type(field_value)))

        if not mask:
            self._mask = [ORIGIN] * len(sentence)
        else:
            assert len(mask) == len(sentence)
            for mask_item in mask:
                if mask_item not in [ORIGIN, TASK_MASK, MODIFIED_MASK]:
                    raise ValueError(
                        "Not support mask value of {0}".format(mask_item))
            self._mask = mask

        super().__init__(sentence, field_type=str)
        self._ner_tags = None
        self._pos_tags = None
        self.token = [k for k in self.field_value]

    def __hash__(self):
        return hash(self.field_value)

    @property
    def mask(self):
        return self._mask[:]

    def ner(self):
        r"""
        ner fiction

        :return: ner tags
        """
        if not self._ner_tags:
            ner_tags, self._ner_list = self.cn_processor.get_ner(
                self.field_value)
            if len(self._ner_list) != len(self.field_value):
                raise ValueError(
                    f"Ner tagging not aligned with tokenized words")
            self._ner_tags = ner_tags
        return self._ner_tags, self._ner_list

    def pos_tags(self):
        r"""
        pos tags fiction

        :return: ner tags
        """
        if not self._pos_tags:
            self._pos_tags = self.cn_processor.get_pos_tag(self.field_value)
        return self._pos_tags
