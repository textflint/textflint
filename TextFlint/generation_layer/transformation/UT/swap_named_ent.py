r"""
SwapNamedEnt substitute class
==========================================================
"""

__all__ = ['SwapNamedEnt']

from ..transformation import Transformation
from ....common.settings import ENTITIES_PATH, CORENLP_ENTITY_MAP
from ....common.utils.load import json_loader
from ....common.utils.list_op import trade_off_sub_words
from ....common.utils.install import download_if_needed


class SwapNamedEnt(Transformation):
    r"""
    Swap entities with other entities of the same category.

    """
    def __init__(
        self,
        entity_res=None,
        **kwargs
    ):
        r"""
        :param dict entity_res: dic of categories and their entities.
        """
        super().__init__()
        entities_path = entity_res if entity_res else download_if_needed(
            ENTITIES_PATH)
        self.entities_dic = json_loader(entities_path)

    def __repr__(self):
        return 'SwapNamedEnt'

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according transform_field.

        :param ~Sample sample: input data, normally one data component.
        :param str field:  indicate which field to transform.
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """
        trans_samples = []
        entities_info = sample.get_ner(field)

        # replace sub strings by contractions
        indices, entities, categories = self.decompose_entities_info(
            entities_info)
        candidates = self._get_random_entities(categories, n)
        candidates, indices = trade_off_sub_words(candidates, indices, n=n)

        if not indices:
            return []

        for i in range(len(candidates)):
            candidate = candidates[i]
            trans_samples.append(
                sample.unequal_replace_field_at_indices(
                    field, indices, candidate))

        return trans_samples

    @staticmethod
    def decompose_entities_info(entities_info):
        r"""
        Decompose given entities and normalize entity tag to ['LOCATION',
        'PERSON', 'ORGANIZATION']

        Example::

            [('Lionel Messi', 0, 2, 'PERSON'),
            ('Argentina', 7, 8, 'LOCATION')]

        >> [[0, 2], [7, 8]], ['Lionel Messi', 'Argentina'],
            ['PERSON', 'LOCATION']

        :param dict entities_info: parsed by default ner component.
        :return list indices: indices
        :return list entities: entity values
        :return list categories: categories

        """
        indices = []
        entities = []
        categories = []

        for entity_info in entities_info:
            category = entity_info[3]
            if category in CORENLP_ENTITY_MAP:
                entities.append(entity_info[0])
                indices.append([entity_info[1], entity_info[2]])
                categories.append(CORENLP_ENTITY_MAP[category])

        return indices, entities, categories

    def _get_random_entities(self, categories, n):
        r"""
        Random generate entities of given categories

        :param list categories:
        :param int n:
        :return list rand_entities: indices and random entities respectively

        """
        rand_entities = []

        for i in range(len(categories)):
            if categories[i] not in self.entities_dic:
                rand_entities.append([])
            else:
                rand_entities.append(self.sample_num(
                    self.entities_dic[categories[i]], n))

        return rand_entities
