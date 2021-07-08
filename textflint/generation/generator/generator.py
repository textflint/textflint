"""
Generator base Class
============================================

"""
__all__ = ["Generator"]

from tqdm import tqdm
from abc import ABC
from itertools import product

from ...input.dataset import Dataset
from ...common.utils.seed import set_seed
from ...common.utils.logger import logger
from ...common.preprocess import EnProcessor
from ...common.utils.load import pkg_class_load, load_module_from_file
from ..transformation import Transformation, Pipeline
from ..subpopulation import SubPopulation
from ...common.settings import TASK_TRANSFORMATION_PATH, \
    ALLOWED_TRANSFORMATIONS, TASK_SUBPOPULATION_PATH, ALLOWED_SUBPOPULATIONS

Flint = {
    "transformation": {
        'task_path': TASK_TRANSFORMATION_PATH,
        'allowed_methods': ALLOWED_TRANSFORMATIONS
    },
    "subpopulation": {
        'task_path': TASK_SUBPOPULATION_PATH,
        'allowed_methods': ALLOWED_SUBPOPULATIONS
    }
}


class Generator(ABC):
    r"""
    Transformation controller which applies multi transformations
    to each data sample.

    """

    def __init__(
        self,
        task='UT',
        max_trans=1,
        random_seed=1,
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
        r"""

        :param str task: Indicate which task of your transformation data.
        :param int max_trans: Maximum transformed samples generate by one
            original sample pre Transformation.
        :param int random_seed: random number seed to reproduce generation.
        :param str|list fields: Indicate which fields to apply transformations.
            Multi fields transform just for some special task, like: SM、NLI.
        :param list trans_methods: list of transformations' name.
        :param dict trans_config: transformation class configs, useful
            to control the behavior of transformations.
        :param bool return_unk: Some transformation may generate unk labels,
            s.t. insert a word to a sequence in NER task.
            If set False, would skip these transformations.
        :param list sub_methods: list of subpopulations' name.
        :param dict sub_config: subpopulation class configs, useful
            to control the behavior of subpopulation.
        :param str attack_methods: path to the python file containing
         the Attack instances.
        :param list validate_methods: confidence calculate functions.
        """
        self.task = task
        self.max_trans = max_trans
        self.random_seed = random_seed
        self.fields = fields

        self.return_unk = return_unk
        # text processor to do nlp preprocess
        self.processor = EnProcessor()
        self.transform_methods = trans_methods
        self.trans_config = trans_config \
            if trans_config else {}

        self.sub_methods = sub_methods
        self.sub_config = sub_config \
            if sub_config else {}
        self.attack_methods = attack_methods
        self.validate_methods = validate_methods

    def prepare(self, dataset):
        r"""
        Check dataset

        :param textflint.Dataset dataset: the input dataset

        """
        assert isinstance(dataset, Dataset)
        self._check_dataset(dataset)
        self._check_fields()
        set_seed(self.random_seed)

    def generate(self, dataset, model=None):
        r"""
        Returns a list of possible generated samples for ``dataset``.

        :param textflint.Dataset dataset: the input dataset
        :param textflint.FlintModel model: the model to attack if given.
        :return: yield (original samples, new samples,
            generated function string).

        """
        for data in self.generate_by_transformations(dataset):
            yield data
        for data in self.generate_by_subpopulations(dataset):
            yield data
        for data in self.generate_by_attacks(dataset, model):
            yield data

    def generate_by_transformations(self, dataset, **kwargs):
        r"""
        Generate samples by a list of transformation methods.

        :param dataset: the input dataset
        :return: (original samples, new samples, generated function string)

        """
        self.prepare(dataset)

        for trans_obj in self._get_flint_objs(
            self.transform_methods,
            TASK_TRANSFORMATION_PATH,
            ALLOWED_TRANSFORMATIONS
        ):
            # initialize current index of dataset
            dataset.init_iter()

            logger.info('******Start {0}!******'.format(trans_obj))
            generated_samples = dataset.new_dataset()
            original_samples = dataset.new_dataset()

            for sample in tqdm(dataset):
                # default return list of samples
                trans_rst = trans_obj.transform(
                    sample, n=self.max_trans, field=self.fields
                )
                if trans_rst:
                    generated_samples.extend(trans_rst)
                    original_samples.append(sample)

            yield original_samples, generated_samples, trans_obj.__repr__()
            logger.info('******Finish {0}!******'.format(trans_obj))

    def generate_by_subpopulations(self, dataset, **kwargs):
        r"""
        Generate samples by a list of subpopulation methods.

        :param dataset: the input dataset
        :return: the transformed dataset

        """
        self.prepare(dataset)

        for sub_obj in self._get_flint_objs(
            self.sub_methods,
            TASK_SUBPOPULATION_PATH,
            ALLOWED_SUBPOPULATIONS
        ):
            logger.info('******Start {0}!******'.format(sub_obj))
            generated_samples = sub_obj.slice_population(dataset, self.fields)
            yield None, generated_samples, sub_obj.__repr__()
            logger.info('******Finish {0}!******'.format(sub_obj))

    def generate_by_attacks(self, dataset, model=None, **kwargs):
        r"""
        Generate samples by a list of attack methods.

        :param dataset: the input dataset
        :param model: the model to attack if given.
        :return: the transformed dataset

        """
        self.prepare(dataset)

        for attack_obj in self._get_attack_objs(
            self.attack_methods, model
        ):
            logger.info('******Start Attack!******')
            logger.info(attack_obj)
            original_samples, generated_samples = attack_obj.attack_dataset(
                dataset)
            yield original_samples, generated_samples, attack_obj.__repr__()
            logger.info('******Finish Attack!******')

    def _get_attack_objs(self, attack_methods, model):
        """
        Get the objects of attack methods.

        :param str attack_methods: path to the python file containing
            the Attack instances.
        :param model: the model to be attacked
        :return: list of objects of attacks.
        """
        if attack_methods:
            attacks = load_module_from_file("attacks", attack_methods)
            for attack in attacks:
                attack.init_goal_function(model)
        else:
            attacks = []
        return attacks

    def _get_flint_objs(self, flint_methods, flint_path, allowed_methods):
        """
        Allow UT transformations and task specific transformations.
        Support instance single transformation and pipeline transformations.

        :param list flint_methods: the method name
        :param dict flint_path: task to its flint objs path
        :param dict allowed_methods: the allowed flint methods of this task
        :return: list of objects of flint_methods.
        """
        flint_objs = []
        flint_type = Transformation if flint_path is TASK_TRANSFORMATION_PATH \
            else SubPopulation
        flint_classes = pkg_class_load(flint_path[self.task], flint_type)
        if self.task != 'UT' and self.task != 'CWS':
            flint_classes.update(pkg_class_load(flint_path['UT'], flint_type))

        if isinstance(flint_methods, list):
            if len(flint_methods) == 0:
                return []
        elif flint_methods:
            raise ValueError(f'The type of trans_methods must be list '
                             f'or None, not {type(flint_methods)}')
        else:
            flint_methods = allowed_methods[self.task]
        for flint_method in flint_methods:
            flint_objs.extend(
                self._create_flint_objs(flint_method, flint_classes, flint_path)
            )
        return flint_objs

    def _create_flint_objs(self, flint_method, flint_classes, flint_path):
        """
        Check and create transform method instance.

        :param str flint_method: flint method string.
        :param dict flint_classes: method to its flint classes
        :param dict flint_path: task to its flint objs path
        :return: flint instances list
        """

        if isinstance(flint_method, str):
            methods = [flint_method]
        elif isinstance(flint_method, list):
            methods = flint_method
        else:
            raise ValueError(
                "Method {0} is not allowed in task {1}".format(
                    flint_method, self.task)
            )
        config = self.trans_config \
            if flint_path is TASK_TRANSFORMATION_PATH \
            else self.sub_config

        objs = []
        # support any legal sequence transformation
        for method in methods:
            new_objs = []
            method_params = config.get(method, {})
            if method in flint_classes:
                if isinstance(method_params, list):
                    # create multi objects according passed params

                    for index in range(len(method_params)):
                        # multi objs not pipeline objs
                        new_objs.append(
                            [flint_classes[method](**method_params[index])])
                else:
                    new_objs.append([flint_classes[method](**method_params)])
            else:
                raise ValueError(
                    "Method {0} is not allowed in task {1}".format(
                        method, self.task)
                )

            if not objs:
                objs = [obj[0] for obj in new_objs] if len(
                    methods) == 1 else new_objs
            else:
                cached_objs = []

                for x in product(objs, new_objs):
                    cached_objs.append(Pipeline(list(x[0] + x[1])))
                objs = cached_objs

        return objs

    def _check_fields(self):
        """
        Check whether fields is legal.

        """
        if isinstance(self.fields[0], str):
            pass
        elif isinstance(self.fields, list):
            if len(self.fields) == 1 and isinstance(self.fields[0], str):
                self.fields = self.fields[0]
            else:
                raise ValueError(
                    'Task {0} not support transform multi fields: {0}'.format
                    (self.task, self.fields)
                )
        else:
            raise ValueError(
                'Task {0} not support input fields type: {0}'.format(
                    self.task, type(self.fields)
                )
            )

    def _check_dataset(self, dataset):
        """
        Check given dataset whether compatible with task and fields.

        :param textflint.dataset.Dataset dataset: the input dataset.
        """
        # check whether empty
        if not dataset or len(dataset) == 0:
            raise ValueError('Input dataset is empty!')
        # check dataset whether compatible with task and fields
        data_sample = dataset[0]

        if self.task.lower() not in data_sample.__repr__().lower():
            raise ValueError(
                'Input data sample type {0} is not compatible with task {1}'
                .format(data_sample.__repr__(), self.task)
            )
        if isinstance(self.fields, str):
            fields = [self.fields]
        else:
            fields = self.fields

        for field_str in fields:
            if not hasattr(data_sample, field_str):
                raise ValueError('Cant find attribute {0} from {1}'
                                 .format(field_str, data_sample.__repr__()))
