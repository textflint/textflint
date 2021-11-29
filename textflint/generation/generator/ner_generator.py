r"""
NER Generator aims to apply NER data generation function
==========================================================
"""
__all__ = ["NERGenerator"]
from tqdm import tqdm

from ...common import logger
from .generator import Generator
from ...input.dataset import Dataset
from ...common.settings import TASK_TRANSFORMATION_PATH, \
    ALLOWED_TRANSFORMATIONS, TASK_SUBPOPULATION_PATH, ALLOWED_SUBPOPULATIONS
Flint = {
    "transformation": {'task_path': TASK_TRANSFORMATION_PATH,
                       'allowed_methods': ALLOWED_TRANSFORMATIONS},
    "subpopulation": {'task_path': TASK_SUBPOPULATION_PATH,
                      'allowed_methods': ALLOWED_SUBPOPULATIONS}
}


class NERGenerator(Generator):
    r"""
    NER Generator aims to apply NER data generation function.

    """
    def __init__(
        self,
        task='NER',
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

    def generate_by_transformations(self, dataset, **kwargs):
        r"""
        Returns a list of all possible transformed samples for "dataset".

        :param ~TextRobustness.dataset.Dataset dataset: the original dataset
            ready for transformation or subpopulation
        :return: yield transformed samples + transformation name string.
        """
        self.prepare(dataset)

        transform_objs = self._get_flint_objs(
            self.transform_methods,
            TASK_TRANSFORMATION_PATH,
            ALLOWED_TRANSFORMATIONS)

        for obj_id, trans_obj in enumerate(transform_objs):
            logger.info('******Start {0}!******'.format(trans_obj))
            generated_samples = dataset.new_dataset()
            original_samples = dataset.new_dataset()
            dataset.init_iter()
            for index in tqdm(range(len(dataset))):
                concat_samples = dataset[index + 1: index + 3]
                sample = dataset[index]
                # default return list of samples
                trans_rst = trans_obj.transform(sample,
                                                n=self.max_trans,
                                                field=self.fields,
                                                concat_samples=concat_samples)
                # default return list of samples
                if trans_rst:
                    generated_samples.extend(trans_rst)
                    original_samples.append(sample)

            yield original_samples, generated_samples, trans_obj.__repr__()
            logger.info('******Finish {0}!******'.format(trans_obj))
            # free transformation object memory
            transform_objs[obj_id] = None
            del trans_obj
