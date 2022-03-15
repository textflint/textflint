r"""
WSCGenerator class for sample generating
"""
__all__ = ["WSCGenerator"]
from .generator import Generator
from ...common.settings import TASK_TRANSFORMATIONS, TASK_TRANSFORMATION_PATH, \
    ALLOWED_TRANSFORMATIONS, TASK_SUBPOPULATION_PATH, ALLOWED_SUBPOPULATIONS

Flint = {
    "transformation": {'task_path': TASK_TRANSFORMATION_PATH,
                       'allowed_methods': ALLOWED_TRANSFORMATIONS},
    "subpopulation": {'task_path': TASK_SUBPOPULATION_PATH,
                      'allowed_methods': ALLOWED_SUBPOPULATIONS}
}


class WSCGenerator(Generator):
    r"""
    generate WSC sample

    """
    def __init__(
        self,
        task='WSC',
        max_trans=1,
        fields='text',
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


