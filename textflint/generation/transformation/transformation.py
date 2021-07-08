"""
Transformation Abstract Class
============================================
"""
__all__ = ["Transformation"]
import random
from abc import ABC, abstractmethod

from ...common.utils import logger
from ...common.utils.list_op import trade_off_sub_words
from ...common.utils.error import FlintError
from ...common.preprocess.en_processor import EnProcessor


class Transformation(ABC):
    r"""
    An abstract class for transforming a sequence of text to produce a list of
    potential adversarial example.

    """
    processor = EnProcessor()

    def __init__(
        self,
        **kwargs
    ):
        pass

    def __repr__(self):
        return 'Transformation'

    def transform(self, sample, n=1, field='x', **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~textflint.input.component.sample.Sample sample: Data
            sample for augmentation.
        :param int n: Max number of unique augmented output, default is 5.
        :param str|list field: Indicate which fields to apply transformations.
        :param dict **kwargs: other auxiliary params.
        :return: list of Sample

        """
        if n < 1:
            return []

        if not isinstance(field, list):
            assert isinstance(field, str), "The type of field must be a str " \
                                           "or list not {0}".format(type(field))
            fields = [field]
        else:
            fields = field

        assert isinstance(fields, list), \
            "The type of field can choice in str or" \
            " list,not {0}".format(type(field))
        fields = list(set(fields))

        try:  # Deal with textflint Exception
            if len(fields) == 1:
                transform_results = self._transform(sample, n=n,
                                                    field=fields[0], **kwargs)
            else:
                transform_results = []
                for field in fields:
                    transform_results.append([[trans.get_value(field),
                                               trans.get_mask(field)]
                                              for trans in self._transform(
                            sample, n=n, field=field, **kwargs)])
                trans_items, trans_fields = trade_off_sub_words(
                    transform_results, fields, n=n)
                transform_results = []
                for trans_item in trans_items:
                    transform_results.append(
                        sample.replace_fields(
                            trans_fields, [k[0] for k in trans_item],
                            field_masks=[k[1] for k in trans_item]))

        except FlintError as e:
            logger.error(str(e))
            return []
        except Exception as e:
            logger.error(str(e))
            raise FlintError("You hit an internal error. "
                             "Please open an issue in "
                             "https://github.com/textflint/textflint"
                             " to report it.")
        if transform_results:
            return [sample for sample in transform_results
                    if (not sample.is_origin) and sample.is_legal()]
        else:
            return []

    @abstractmethod
    def _transform(self, sample, n=1, field='x', **kwargs):
        r"""
        Returns a list of all possible transformations for ``component``.

        :param ~textflint.input.component.sample.Sample sample:
            Data sample for augmentation.
        :param int n: Default is 5. MAx number of unique augmented output.
        :param str field: Indicate which field to apply transformations.
        :param dict **kwargs: other auxiliary params.
        :return: list of Sample

        """
        raise NotImplementedError

    @classmethod
    def sample_num(cls, x, num):
        r"""
        Get 'num' samples from x.

        :param list x: list to sample
        :param int num: sample number
        :return: max 'num' unique samples.

        """
        if isinstance(x, list):
            num = min(num, len(x))
            return random.sample(x, num)
        elif isinstance(x, int):
            num = min(num, x)
            return random.sample(range(0, x), num)
