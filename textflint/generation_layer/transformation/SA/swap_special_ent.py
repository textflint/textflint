"""
Word Swap by swapping names according to the person or movies in the sentence from csv file
==========================================================
"""

__all__ = ["SwapSpecialEnt"]

import random
from ..transformation import Transformation
from ....common.settings import SA_PERSON_PATH, SA_MOVIE_PATH
from ....common.utils.install import download_if_needed
from ....common.utils.load import sa_dict_loader


class SwapSpecialEnt(Transformation):
    r"""
    Transforms an input by adding summaries of person and movies provided
    by csv.

    Example::
        ori: Titanic is my favorite movie
        trans: The Boys Club is my favorite movie

    """

    def __init__(
        self,
        entity_type='person',
        **kwargs
    ):
        r"""

        :param str entity_type: entity for which entity type
        :param kwargs: other params

        """

        super().__init__()
        self.entity_type = entity_type
        if entity_type == 'movie':
            self.entity_dict, self.max_entity_len = \
                sa_dict_loader(download_if_needed(SA_MOVIE_PATH))
        elif entity_type == 'person':
            self.entity_dict, self.max_entity_len = \
                sa_dict_loader(download_if_needed(SA_PERSON_PATH))
        else:
            raise ValueError(
                'SpecialEntityReplace not support type {0}, '
                'please choose entity type from movie and person'.format(
                    entity_type))

    def __repr__(self):
        return 'SwapSpecialEnt' + '-' + self.entity_type

    def _transform(self, sample, field='x', n=5, **kwargs):
        r"""
        Transform text string according field.

        :param ~SASample sample: input data, a SASample contains
            'x' field and 'y' field
        :param string field: indicate which fields to transform,
            for multi fields, substitute them at the same time.
        :param int n: number of generated samples
        :return list trans_samples: transformed sample list.
        """
        # To speed up the query, dividing the original sentence into n-tuple
        # string
        tup_list = sample.concat_token(self.max_entity_len)
        replace_indices = self._get_entity_location(tup_list)

        if not replace_indices:
            return []

        trans_samples = []

        for i in range(n):
            # Randomly select a name for each candidate position
            sub_entity = random.sample(
                self.entity_dict.keys(), len(replace_indices))
            sub_entity_token = [self.processor.tokenize(entity)
                                for entity in sub_entity]
            trans_samples.append(sample.unequal_replace_field_at_indices(
                'x', replace_indices, sub_entity_token))
        return trans_samples

    def _get_entity_location(self, tup_list):
        r"""
        Get the indices of the words to be replaced and the new names to
        replace them

        :param list tup_list: a n-tuple list to speed up searching
        :return list indices: indices of tokens that should be replaced
        :return list names: The names that correspond to indices and is
            used to replace them
        """
        def check_collision(index, r):
            for i, range1 in enumerate(r):
                l1, r1 = range1
                l2, r2 = index
                if max(l1, l2) < min(r1, r2):
                    return True
            return False

        candidates_indices = []

        for item in tup_list:
            current_str = item['string']
            current_indices = item['indices']

            if current_str in self.entity_dict and not check_collision(
                    current_indices, candidates_indices):
                candidates_indices.append(current_indices)

        return candidates_indices


