# -- coding: utf-8 --
r"""
Substitute short entities to longer ones
==========================================================
"""
__all__ = ["SwapEnt"]
import random

from ....common.settings import CN_LONG_ENTITIES
from ....common.settings import CN_OOV_ENTITIES, LABEL_TRANS
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
        if swap_type not in ['OOV', 'SwapLonger']:
            raise ValueError(
                'Not support {0} type, plz ensure swap_type in {1}' .format(
                    swap_type, ['OOV', 'SwapLonger']))
        self.swap_type = swap_type
        
        if swap_type == 'SwapLonger':
            res_path = download_if_needed(CN_LONG_ENTITIES) if not res_path else \
                res_path
            self.res_dic = json_loader(res_path)


        elif swap_type == 'OOV':
            res_path = download_if_needed(CN_OOV_ENTITIES) if not res_path else \
                res_path
            self.res_dic = json_loader(res_path)


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
        entities = sample.entities
        candidates = []

        for entity in entities:
            assert entity['tag'] in ['PER','LOC','GPE','ORG','NS','NT','NR'], \
                '{0} is not supported'.format(entity['tag'])

            if self.swap_type=='SwapLonger' and entity['entity'] in self.res_dic:
               
                rep_entities.append(entity)
                candidates.append(self.res_dic[entity['entity']])
            
            elif self.swap_type=='OOV':
                
                trans_tag = LABEL_TRANS[entity['tag']]
                
                rep_entities.append(entity)
                candidates.append(random.sample(self.res_dic[trans_tag], n)[0])

        for i in range(n):
            _candidates = [candidate for candidate in candidates]
            if not _candidates:
                return []
            rep_samples.append(sample.entities_replace(
                rep_entities, _candidates))

        return rep_samples


