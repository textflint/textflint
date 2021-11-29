r"""
Apply sequential transformations to input sample.
Default generate transformed samples of combination number of contained transformations.
==========================================================
"""
__all__ = ["Pipeline"]

from ..transformation import Transformation


class Pipeline(Transformation, list):
    r"""
    Apply sequential transformations to input sample.
    Default generate transformed samples of combination number of contained
    transformations.

    """

    def __init__(
        self,
        transform_objs
    ):
        Transformation.__init__(self)
        list.__init__(self, [])

        if not isinstance(transform_objs, list):
            transform_objs = [transform_objs]

        # add unique legal transformations
        for transform_obj in transform_objs:
            self.append(transform_obj)

    def __repr__(self):
        return 'Pipeline_' + "_".join([format(trans) for trans in self[:]])

    def _transform(self, sample, n=1, field='x', **kwargs):
        r"""
        Returns samples of combination number of contained transformations..

        :param ~textflint.input.component.sample.Sample sample:
            Data sample for augmentation.
        :param int n: Default is 5. Max number of unique augmented output.
        :param str|list: Correspond with transform_objs,
            and indicate which field to apply transformations.
        :param dict kwargs: Other auxiliary params.
        :return: list of Sample

        """
        trans_samples = [sample]
        if isinstance(field, str):
            fields = [field] * len(self)
        elif isinstance(field, list):
            assert len(field) == len(self)
            fields = field
        else:
            raise ValueError('Do not support field input {0}'.format(field))

        for index in range(len(self)):
            cached_samples = []

            for trans_sample in trans_samples:
                trans_result = self[index].transform(
                    trans_sample, n=n, field=fields[index], **kwargs)

                if trans_result:
                    cached_samples.extend(trans_result)

            # sample n result from n^2 trans results
            trans_samples = self.sample_num(cached_samples, n)

        return trans_samples

    def get_transformations(self):
        r"""
        :return: List of transformation string.

        """
        return [str(trans_obj) for trans_obj in self]
