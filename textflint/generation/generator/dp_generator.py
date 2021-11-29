r"""
DP Generator Class
============================================

"""

from .generator import Generator
from ...input.component.sample.dp_sample import DPSample

__all__ = ["DPGenerator"]


class DPGenerator(Generator):
    r"""
    Dependency Parsing Generator aims to
    apply Dependency Parsing data generation function.

    """
    def __init__(
        self,
        task='DP',
        max_trans=1,
        fields='x',
        trans_methods=None,
        trans_config=None,
        return_unk=True,
        sub_methods=None,
        sub_config=None,
        attack_methods=None,
        validate_methods=None,
        **kwargs
    ):
        super().__init__(
            task=task,
            max_trans=max_trans,
            fields=fields,
            trans_methods=trans_methods,
            trans_config=trans_config,
            return_unk=return_unk,
            sub_methods=sub_methods,
            sub_config=sub_config,
            attack_methods=attack_methods,
            validate_methods=validate_methods,
            **kwargs
        )
