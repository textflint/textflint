r"""
Perturb Answer with BackTrans or MLM
==========================================================
"""
__all__ = ['PerturbQuestion']

from ..transformation import Transformation
from ...transformation.UT import BackTrans, MLMSuggestion


class PerturbQuestion(Transformation):
    r"""
    Transform the question

    Example::

        origin: Where was Super Bowl 50 held?
        transform: Where did Super Bowl 50 take place?
    """

    def __init__(
            self,
            transform_method='BackTrans',
            device="cuda:0"):
        r"""
        :param transform_method: paraphrase method
        :param device: GPU device or CPU
        """
        super().__init__()

        # Rules for altering sentences
        self.transform_method = transform_method
        if transform_method == 'BackTrans':
            self.tranf = BackTrans(device=device)
        else:
            self.tranf = MLMSuggestion(device=device)

    def __repr__(self):
        return 'PerturbQuestion' + '-' + self.transform_method

    def _transform(self, sample, **kwargs):
        r"""
        Paraphrase the question with BackTrans or MLM
        :param sample: the sample to transform
        :param kwargs:
        :return: list of sample
        """

        transform_samples = []
        new_sample = self.tranf.transform(sample=sample, field='question', n=1)
        transform_samples.extend(new_sample)
        return transform_samples
