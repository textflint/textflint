"""
textflint Adapter Class
============================================

"""

import os

from .input_layer.config import Config
from .common import logger
from .input_layer.dataset import Dataset
from .input_layer.model import FlintModel
from .common.utils import task_class_load
from .generation_layer.generator import Generator, UTGenerator
from .common.settings import GENERATOR_PATH, NLP_TASK_MAP
from .report_layer.report_generator import ReportGenerator

nlp_tasks = [key.upper() for key in NLP_TASK_MAP.keys()]

__all__ = ["Adapter"]


class Adapter:
    @staticmethod
    def get_config(task='UT', config=None):
        r"""
        Check config input or create config automatically.

        :param str task: task name
        :param str|dict|textflint.config.Config config: config to control
            generation procedure.
        :return: textflint.config.Config instance.

        """

        if config is None:
            config_obj = Config(task=task)
        elif isinstance(config, dict):
            config_obj = Config(**config)
        elif isinstance(config, str):
            assert os.path.exists(config), f"Cant load config from {config}"
            config_obj = Config().from_json_file(config)
        elif isinstance(config, Config):
            config_obj = config
        else:
            raise ValueError("Invalid config type {0}!".format(type(config)))

        return config_obj

    @staticmethod
    def get_generator(config_obj):
        r"""
        Automatic create task generator to apply transformations, subpopulations
        and adversarial attacks.

        :param textflint.Config config_obj: Config instance.
        :return: textflint.Generator

        """
        # get references of different nlp task Configs
        assert isinstance(config_obj, Config), \
            f'Cant initialize generator with {config_obj}'
        generator_map = task_class_load(GENERATOR_PATH, nlp_tasks, Generator,
                                        filter_str='_generator')
        task = config_obj.task

        if task.upper() not in generator_map:
            logger.warning(f'Do not support task: {task}, '
                           f'default utilize UT generator.')
            generator_obj = UTGenerator(**config_obj.to_dict())
        else:
            generator_obj = generator_map[task.upper()](**config_obj.to_dict())

        return generator_obj

    @staticmethod
    def get_dataset(data_input=None, task='UT'):
        r"""
        Create Dataset instance and load data input automatically.

        :param dict|list|string data_input: json object or json/csv file.
        :param str task: task name.
        :return: textflint.Dataset instance.

        """
        def split_huggingface_data_str(data_str):
            raise NotImplementedError

        if not isinstance(data_input, (list, dict, str)):
            raise ValueError('Please pass a dataset dic, or local csv path, '
                             'or HuggingFace data str, your input is {0}'
                             .format(data_input))
        dataset_obj = Dataset(task=task)

        if isinstance(data_input, (list, dict)):
            dataset_obj.load(data_input)
        else:
            if os.path.exists(data_input):
                if '.csv' in data_input:
                    dataset_obj.load_csv(csv_path=data_input)
                elif '.json' in data_input:
                    dataset_obj.load_json(json_path=data_input)
                else:
                    raise ValueError('Not support load {0}, please ensure '
                                     'your input file end with \'.csv\' '
                                     'or \'.json\''.format(data_input))
            else:
                raise ValueError('Just support load json object, '
                                 'csv file and json file, error input {0}'
                                 .format(data_input))

        return dataset_obj

    @staticmethod
    def get_flintmodel(model=None, task=None):
        r"""
        Check flint model type and whether compatible to task.

        :param model:
        :return:
        """
        assert isinstance(model, FlintModel), f"Not support {type(model)} " \
            f"input, please wrapper your model with FlintModel."
        assert model.task == task, f"The task of your FlintModel is " \
            f"{model.task}, not compatible with task {task}"

        return model

    @staticmethod
    def get_report_generator():
        r"""
        Return a ReportGenerator instance.

        :return: ReportGenerator

        """
        return ReportGenerator()
