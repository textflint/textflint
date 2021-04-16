r"""
 Generator aims to apply NER data generation function
==========================================================
"""
__all__ = ["CWSGenerator"]
from .generator import Generator


class CWSGenerator(Generator):
    r"""
    CWS Generator aims to apply CWS data generation function.

    """
    def __init__(
        self,
        task='CWS',
        fields='x',
        max_trans=1,
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
            fields=fields,
            max_trans=max_trans,
            transformation_methods=transformation_methods,
            transformation_config=transformation_config,
            return_unk=return_unk,
            subpopulation_methods=subpopulation_methods,
            subpopulation_config=subpopulation_config,
            attack_methods=attack_methods,
            validate_methods=validate_methods,
            **kwargs
        )

