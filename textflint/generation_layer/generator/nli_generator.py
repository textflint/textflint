r"""
NLI Generator Class
============================================

"""
__all__ = ['NLIGenerator']

from .generator import Generator


class NLIGenerator(Generator):
    def __init__(
            self,
            task='NLI',
            max_trans=1,
            fields=['premise', 'hypothesis'],
            transformation_methods=None,
            transformation_config=None,
            return_unk=True,
            subpopulation_methods=None,
            subpopulation_config=None,
            attack_methods=None,
            validate_methods=None
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
            validate_methods=validate_methods
        )

    def _check_fields(self):
        r"""
        Check whether fields is legal.

        :return:
        """
        if isinstance(self.fields, str):
            self.fields = [self.fields]
        elif isinstance(self.fields, list):
            pass
        else:
            raise ValueError(
                f'Task {self.task} not support input fields '
                f'type: {type(self.fields)}'
            )

        for field in self.fields:
            if field not in ['premise', 'hypothesis', 'y']:
                raise ValueError(
                    f'Task {self.task} not support input fields: {field}'
                )
