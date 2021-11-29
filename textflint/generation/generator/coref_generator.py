from ...input.component.sample import CorefSample
from .generator import Generator
from ...common.utils.logger import logger
from ...common.settings import TASK_TRANSFORMATION_PATH, \
    ALLOWED_TRANSFORMATIONS, TASK_SUBPOPULATION_PATH, ALLOWED_SUBPOPULATIONS
from tqdm import tqdm
import random


Flint = {
    "transformation": {'task_path': TASK_TRANSFORMATION_PATH,
                       'allowed_methods': ALLOWED_TRANSFORMATIONS},
    "subpopulation": {'task_path': TASK_SUBPOPULATION_PATH,
                      'allowed_methods': ALLOWED_SUBPOPULATIONS}
}


class CorefGenerator(Generator):
    r"""
    Generate new samples for coref task.

    """
    def __init__(
        self,
        task='COREF',
        max_trans=1,
        fields='x',
        trans_methods=None,
        trans_config=None,
        return_unk=True,
        sub_methods=None,
        sub_config=None,
        attack_methods=None,
        validate_methods=None,
        num_other_samples=2,
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
        self.num_other_samples = num_other_samples # default 2

    def generate_by_transformations(self, dataset, **kwargs):
        r"""
        Returns a list of all possible transformed samples for ``dataset``.

        :param ~dataset dataset: dataset
        :return: yield transformed samples
        """
        self.prepare(dataset)

        dataset_ls = dataset.dump()
        transform_objs = self._get_flint_objs(
            self.transform_methods,
            TASK_TRANSFORMATION_PATH,
            ALLOWED_TRANSFORMATIONS)

        for obj_id, trans_obj in enumerate(transform_objs):
            logger.info('******Start {0}!******'.format(trans_obj))
            generated_samples = dataset.new_dataset()
            original_samples = dataset.new_dataset()
            # initialize current index of dataset
            dataset.init_iter()

            for i, sample in enumerate(tqdm(dataset_ls)):
                if isinstance(sample, dict):
                    sample = CorefSample(sample)
                if len(dataset_ls) <= self.num_other_samples:
                    samples_other = dataset_ls[:i] + dataset_ls[i + 1:]
                else:
                    samples_other = []
                    while len(samples_other) < self.num_other_samples:
                        rand_idx = random.randint(0, len(dataset_ls)-1)
                        if rand_idx != i:
                            samples_other.append(dataset_ls[rand_idx])
                if len(samples_other) > 0 and isinstance(samples_other[0], dict):
                    samples_other = [CorefSample(s) for s in samples_other]
                trans_rst = trans_obj.transform(
                    sample, n=self.max_trans, field=self.fields,
                    samples_other=samples_other)
                if trans_rst:
                    generated_samples.extend(trans_rst)
                    original_samples.append(sample)

            yield original_samples, generated_samples, trans_obj.__repr__()
            logger.info('******Finish {0}!******'.format(trans_obj))
