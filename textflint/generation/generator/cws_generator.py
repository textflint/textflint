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
            fields=fields,
            max_trans=max_trans,
            trans_methods=trans_methods,
            trans_config=trans_config,
            return_unk=return_unk,
            sub_methods=sub_methods,
            sub_config=sub_config,
            attack_methods=attack_methods,
            validate_methods=validate_methods,
            **kwargs
        )

