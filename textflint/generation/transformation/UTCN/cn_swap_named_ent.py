r"""
SwapNamedEnt substitute class
==========================================================
"""

__all__ = ['CnSwapNamedEnt']

from ..transformation import Transformation
from ....common.settings import CN_CORENLP_ENTITY_MAP, CN_NAME_PATH, CN_LOC_PATH, CN_ORG_PATH
from ....common.utils.load import json_loader
from ....common.utils.list_op import trade_off_sub_words
from ....common.utils.install import download_if_needed


class CnSwapNamedEnt(Transformation):
    r"""
    Swap entities with other entities of the same category.

    """
    def __init__(
            self,
            trans_min=1,
            trans_max=10,
            trans_p=0.1,
            stop_words=None,
            entity_res=None,
            **kwargs
    ):
        r"""
        :param dict entity_res: dic of categories and their entities.
        """
        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words,
        )
        name_path = download_if_needed(CN_NAME_PATH)
        loc_path = download_if_needed(CN_LOC_PATH)
        org_path = download_if_needed(CN_ORG_PATH)
        self.entities_dic = {
            'PERSON': json_loader(name_path)['name'],
            'LOCATION': json_loader(loc_path),
            'ORGANIZATION': json_loader(org_path),
        }

    def __repr__(self):
        return 'CnSwapNamedEnt'

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
        indices, entities, categories = self.decompose_entities_info(entities_info, sample.get_text(field))
        candidates = self._get_random_entities(categories, n)
        candidates, indices = trade_off_sub_words(candidates, indices, n=n)

        if not indices:
            return []

        for i in range(len(candidates)):
            candidate = candidates[i]
            trans_samples.append(
                sample.unequal_replace_field_at_indices(field, indices, candidate))

        return trans_samples

    @staticmethod
    def decompose_entities_info(entities_info, text):
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

        for entity_info in entities_info[0]:
            category = entity_info[0]
            start = entity_info[1]
            end = entity_info[2] + 1
            if category in CN_CORENLP_ENTITY_MAP:
                entities.append(text[start: end])
                indices.append([start, end])
                categories.append(CN_CORENLP_ENTITY_MAP[category])

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
