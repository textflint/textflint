"""
WSD Generator Class
============================================

"""
__all__ = ["WSDGenerator"]

from .generator import Generator
from tqdm import tqdm
from ...common.utils.logger import logger
from ...common.settings import TASK_TRANSFORMATION_PATH, \
    ALLOWED_TRANSFORMATIONS, TASK_SUBPOPULATION_PATH, \
    ALLOWED_SUBPOPULATIONS

Flint = {
    "transformation": {'task_path': TASK_TRANSFORMATION_PATH,
                       'allowed_methods': ALLOWED_TRANSFORMATIONS},
    "subpopulation": {'task_path': TASK_SUBPOPULATION_PATH,
                      'allowed_methods': ALLOWED_SUBPOPULATIONS}
}


class WSDGenerator(Generator):
    def __init__(
            self,
            task='WSD',
            max_trans=1,
            fields='sentence',
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

        :param ~dataset: the original dataset
        :return: yield original samples, new samples, generated function string
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
            # initialize current index of dataset
            dataset.init_iter()

            for sample in tqdm(dataset):
                # default return list of samples
                trans_rst = trans_obj.transform(
                    sample,
                    n=self.max_trans,
                    field=self.fields,
                    split_by_space=True,
                )
                if trans_rst:
                    generated_samples.extend(trans_rst)
                    original_samples.append(sample)

            yield original_samples, generated_samples, trans_obj.__repr__()
            logger.info('******Finish {0}!******'.format(trans_obj))
