r"""
Substitute short entities to longer ones
==========================================================
"""
__all__ = ["SwapEnt"]
import random

from ....common.settings import LONG_ENTITIES
from ....common.settings import CROSS_ENTITIES
from ....common.utils.load import read_cross_entities
from ....common.settings import NER_OOV_ENTITIES
from ....common.utils.load import load_oov_entities
from ....common.utils.install import download_if_needed
from ....common.utils.load import json_loader
from ..transformation import Transformation


class SwapEnt(Transformation):
    r"""
    Swap entities which shorter than threshold to longer ones.

    """

    def __init__(self, swap_type='OOV', res_path=None, **kwargs):
        r"""
        :param string swap_type: the swap type in
        ['CrossCategory', 'OOV', 'SwapLonger']
        
        :param string res_path: dir for vocab/dict
        """
        super().__init__()
        if swap_type not in ['CrossCategory', 'OOV', 'SwapLonger']:
            raise ValueError(
                'Not support {0} type, plz ensure swap_type in {1}' .format(
                    swap_type, ['CrossCategory', 'OOV', 'SwapLonger']))
        self.swap_type = swap_type
        if swap_type == 'SwapLonger':
            res_path = download_if_needed(LONG_ENTITIES) if not res_path else \
                res_path
            self.res_dic = json_loader(res_path)
        elif swap_type == 'CrossCategory':
            res_path = CROSS_ENTITIES if not res_path else res_path
            self.res_dic = read_cross_entities(download_if_needed(res_path))
        else:
            self.res_dic = load_oov_entities(
                download_if_needed(NER_OOV_ENTITIES))

    def __repr__(self):
        return self.swap_type

    def _transform(self, sample, n=1, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~NERSample sample: Data sample for augmentation
        :param int n: Default is 1. MAx number of unique augmented output
        :param **kwargs:
        :return: Augmented data
        """
        rep_samples = []
        rep_entities = []
        entities = sample.entities[::-1]
        candidates = []

        for entity in entities:
            if entity['entity'] in self.res_dic[entity['tag']]:
                res_dic = self.res_dic
                res_dic[entity['tag']].remove(entity['entity'])
            else:
                res_dic = self.res_dic
            if entity['tag'] == "PER" or entity['tag'] == "ORG" or \
                    entity['tag'] == "LOC":
                rep_entities.append(entity)
                candidates.append(random.sample(res_dic[entity['tag']], n))

        for i in range(n):
            _candidates = [candidate[i] for candidate in candidates]
            if not _candidates:
                return []
            rep_samples.append(sample.entities_replace(
                rep_entities, _candidates))

        return rep_samples


