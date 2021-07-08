r"""
MRC Generator Class
============================================

"""
__all__ = ['MRCGenerator']
from .generator import Generator
from tqdm import tqdm
from ...common.utils.logger import logger
from ...common.utils import load_supporting_file
from ...common.settings import NEAR_DICT_PATH, POS_DICT_PATH
from ...common.utils.install import download_if_needed
from ...common.settings import TASK_TRANSFORMATION_PATH, \
    ALLOWED_TRANSFORMATIONS, TASK_SUBPOPULATION_PATH, ALLOWED_SUBPOPULATIONS


class MRCGenerator(Generator):

    def __init__(
        self,
        task='MRC',
        max_trans=1,
        fields='context',
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

        self.nearby_word_dict, self.pos_tag_dict = load_supporting_file(
            download_if_needed(NEAR_DICT_PATH),
            download_if_needed(POS_DICT_PATH))

    def generate_by_transformations(self, dataset, **kwargs):
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
                    sample, n=self.max_trans, field=self.fields,
                    nearby_word_dict=self.nearby_word_dict,
                    pos_tag_dict=self.pos_tag_dict
                )
                if trans_rst:
                    generated_samples.extend(trans_rst)
                    original_samples.append(sample)

            yield original_samples, generated_samples, trans_obj.__repr__()
            logger.info('******Finish {0}!******'.format(trans_obj))
