from ...input_layer.component.sample import CorefSample
from .generator import Generator
from ...common.utils.logger import logger
from ...common.settings import TASK_TRANSFORMATION_PATH, \
    ALLOWED_TRANSFORMATIONS, TASK_SUBPOPULATION_PATH, ALLOWED_SUBPOPULATIONS
from tqdm import trange


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
            validate_methods=validate_methods,
        )

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

            for i in trange(len(dataset_ls)):
                sample = dataset_ls[i]
                if isinstance(sample, dict):
                    sample = CorefSample(sample)
                samples_other = dataset_ls[:i] + dataset_ls[i + 1:]
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
