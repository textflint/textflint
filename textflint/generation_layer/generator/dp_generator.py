r"""
DP Generator Class
============================================

"""

from .generator import Generator
from ...input_layer.component.sample.dp_sample import DPSample

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
        transformation_methods=None,
        transformation_config=None,
        return_unk=True,
        subpopulation_methods=None,
        subpopulation_config=None,
        attack_methods=None,
        validate_methods=None,
        **kwargs
    ):
        super().__init__(
            task=task,
            max_trans=max_trans,
            fields=fields,
            transformation_methods=transformation_methods,
            transformation_config=transformation_config,
            return_unk=return_unk,
            subpopulation_methods=subpopulation_methods,
            subpopulation_config=subpopulation_config,
            attack_methods=attack_methods,
            validate_methods=validate_methods,
            **kwargs
        )
