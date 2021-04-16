"""
textflint Config Class
============================================

"""
__all__ = ["Config"]
import os
import six
import json
import copy

from ...common.utils.logger import logger
from ...common.settings import NLP_TASK_MAP, \
    ALLOWED_TRANSFORMATIONS, TRANSFORM_FIELDS, \
    ALLOWED_SUBPOPULATIONS, ALLOWED_VALIDATORS


class Config:
    r"""
    Hold some config params to control generation and report procedure.

    """

    def __init__(
        self,
        task='UT',
        out_dir=None,
        max_trans=1,
        random_seed=1,
        fields=None,
        flint_model=None,
        trans_methods=None,
        trans_config=None,
        return_unk=True,
        sub_methods=None,
        sub_config=None,
        attack_methods=None,
        validate_methods=None,
        **kwargs
    ):
        """
        :param str task: task name
        :param string out_dir: out dir for saving generated samples,
            default current path.
        :param int max_trans: maximum transformed samples generate by
            one original sample pre Transformation.
        :param int random_seed: random number seed to reproduce generation.
        :param str|list[str] fields: fields on which new samples are generated.
        ::param str model_file: path to the python file containing
         the FlintModel instance which named 'model'.
        :param list trans_methods: indicate what transformations
            to apply to dataset.
        :param dict trans_config: parameters for the initialization
            of the transformation instances.
        :param bool return_unk: whether apply transformations which may
            influence label of sample.
        :param list sub_methods: indicate what subpopulations
            to apply to dataset.
        :param dict sub_config: parameters for the initialization
            of the subpopulation instances.
        :param str attack_methods: path to the python file containing
         the Attack instances which named "attacks".
        :param str|list[str] validate_methods: indicate use which validate
            methods to calculate confidence of generated samples.

        """
        self.task = task
        self.out_dir = out_dir if out_dir else '.'
        self.max_trans = max_trans
        self.fields = fields if fields else TRANSFORM_FIELDS[self.task]
        self.flint_model = flint_model
        self.random_seed = random_seed

        self.trans_methods = \
            self.get_generate_methods(trans_methods,
                                      ALLOWED_TRANSFORMATIONS,
                                      allow_pipeline=True)
        self.trans_config = trans_config \
            if trans_config else {}
        # TODO, support the function. default not return origin and return unk
        self.return_unk = return_unk

        self.sub_methods = \
            self.get_generate_methods(sub_methods,
                                      ALLOWED_SUBPOPULATIONS)
        self.sub_config = sub_config \
            if sub_config else {}

        self.attack_methods = attack_methods
        self.validate_methods = self.get_generate_methods(validate_methods,
                                                          ALLOWED_VALIDATORS)
        self.check_config()

    def check_config(self):
        r"""
        Check common config params.

        """
        if self.task.upper() not in NLP_TASK_MAP:
            logger.error('Your task is {0}, just support {1}.'.format(
                    self.task, NLP_TASK_MAP.keys())
            )

        assert isinstance(self.out_dir, str)
        assert isinstance(self.max_trans, int)
        assert isinstance(self.random_seed, int)
        assert isinstance(self.fields, (str, list))
        assert isinstance(self.trans_config, dict)
        assert isinstance(self.return_unk, bool)
        assert isinstance(self.sub_config, dict)

        if self.flint_model:
            assert os.path.exists(self.flint_model), \
                "Please input a exist python file path " \
                "which contains FlintModel instance"

        if self.attack_methods:
            assert os.path.exists(self.attack_methods), \
                "Please input a exist python file path " \
                "which contains Attack instance"

        if self.validate_methods:
            assert isinstance(self.validate_methods, (str, list))

        # if self.return_unk is True:
        #     logger.info(
        #         'Out label contains UNK label, '
        #         'maybe you need to adjust your evaluate functions.')

    def get_generate_methods(self, methods, task_to_methods,
                             allow_pipeline=False):
        r"""
        Validate transformation or subpopulation methods.

        Watch out!
        Some UT transformations/subpopulations may not compatible with
        your task, please choose your method carefully.

        :param list methods: transformation or subpopulation need to apply
            to dataset. If not provide, return default generated methods.
        :param dict task_to_methods: map allowed methods by task name.
        :param bool allow_pipeline: whether allow pipeline input
        :return: list of transformation/subpopulation.

        """
        allowed_methods = task_to_methods[self.task]
        legal_methods = []

        if methods:
            for method in methods:
                if not isinstance(method, (str, list)):
                    raise ValueError(
                        f'Do not support transformation/subpopulation '
                        f'input type {type(method)}'
                    )

                if isinstance(method, str):
                    if method not in allowed_methods:
                        logger.warning(
                            'Do not support {0}, skip this '
                            'input method'.format(method))
                    else:
                        legal_methods.append(method)
                else:
                    if not allow_pipeline:
                        raise ValueError(
                            f'Do not support pipeline method input {method}'
                        )
                    allow = True

                    for _method in method:
                        if _method not in allowed_methods:
                            logger.warning(
                                'Do not support {0}, skip '
                                'this method'.format(method)
                            )
                            allow = False
                    if allow:
                        legal_methods.append(method)
        else:
            legal_methods = legal_methods + allowed_methods

        return legal_methods

    @classmethod
    def from_dict(cls, json_object):
        r"""
        Constructs a `Config` from a Python dictionary of parameters.

        """
        config = cls(task=json_object['task'])
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        r"""
        Constructs a `Config` from a json file of parameters.

        """
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        r"""
        Serializes this instance to a Python dictionary.

        """
        output = copy.deepcopy(self.__dict__)

        return output

    def to_json_string(self):
        r"""
        Serializes this instance to a JSON string.

        """

        return json.dumps(
            self.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False
        )

    def to_json_file(self, json_file):
        r"""
        Serializes this instance to a JSON file.

        """
        with open(json_file, "w+", encoding='utf-8') as writer:
            json.dump(
                self.to_dict(),
                writer,
                indent=2,
                ensure_ascii=False
            )
