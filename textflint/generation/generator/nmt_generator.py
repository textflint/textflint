r"""
NMT Generator Class
============================================

"""
__all__ = ["NMTGenerator"]
from .generator import Generator
from tqdm import tqdm
from ...common.utils.logger import logger
from ...common.settings import TASK_TRANSFORMATION_PATH, \
    ALLOWED_TRANSFORMATIONS, TASK_SUBPOPULATION_PATH, ALLOWED_SUBPOPULATIONS, TASK_TRANSFORMATIONS

PARALLEL_TRANSFORMATIONS = ['WordCase_upper', 'WordCase_lower', 'ParallelTwitterType_random',
                            'ParallelTwitterType_at', 'ParallelTwitterType_url', 
                            'SwapParallelNum', 'SwapParallelSameWord']

class NMTGenerator(Generator):
    def __init__(
        self,
        task='NMT',
        max_trans=1,
        fields=['source', 'target'],
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
            if str(trans_obj) in PARALLEL_TRANSFORMATIONS:
                transformed_field = self.fields
            else:
                transformed_field = self.fields[0]
            logger.info('******Start {0}!******'.format(trans_obj))
            generated_samples = dataset.new_dataset()
            original_samples = dataset.new_dataset()

            for sample in tqdm(dataset):
                # default return list of samples
                trans_rst = trans_obj.transform(
                    sample, n=self.max_trans, field=transformed_field
                )
                if trans_rst:
                    generated_samples.extend(trans_rst)
                    original_samples.append(sample)

            yield original_samples, generated_samples, trans_obj.__repr__()
            logger.info('******Finish {0}!******'.format(trans_obj))

