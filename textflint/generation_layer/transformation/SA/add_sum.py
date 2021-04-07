r"""
Add summaries according to the person or movies in the sentence from csv file
==========================================================
"""

__all__ = ["AddSum"]

from ..transformation import Transformation
from ....common.settings import SA_PERSON_PATH, SA_MOVIE_PATH
from ....common.utils.install import download_if_needed
from ....common.utils.load import sa_dict_loader


class AddSum(Transformation):
    r"""
    Transforms an input by adding summaries of person and movies provided
    by csv.

    Example::

        ori: Titanic is my favorite movie.
        trans: Titanic(A seventeen-year-old aristocrat falls in love with a
        kind but poor artist aboard the
               luxurious,ill-fated R.M.S. Titanic.) is my favorite movie.
    """

    def __init__(
        self,
        entity_type='person',
        **kwargs
    ):
        r"""
        init AddEntitySummary Class

        :param string entity_type: add summary for which entity type
        """
        super().__init__()
        self.entity_type = entity_type
        if entity_type == 'movie':
            self.entity_dict, self.max_entity_len = sa_dict_loader(
                download_if_needed(SA_MOVIE_PATH))
        elif entity_type == 'person':
            self.entity_dict, self.max_entity_len = sa_dict_loader(
                download_if_needed(SA_PERSON_PATH))
        else:
            raise ValueError(
                'AddEntitySummary not support type {0}, please choose entity '
                'type from movie and person'.format(
                    entity_type))

    def __repr__(self):
        return 'AddSum' + '-' + self.entity_type

    def _transform(self, sample, n=1, **kwargs):
        r"""
        Transform text string, this kind of transformation can only produce
        one sample.

        :param ~SASample sample: input data, a SASample contains 'x' field and
            'y' field
        :param int n: number of generated samples, this transformation can only
            generate one sample
        :return list trans_samples: transformed sample list that only contain
            one sample

        """
        # To speed up the query, dividing the original sentence into n-tuple
        # string
        tup_list = sample.concat_token(self.max_entity_len)
        insert_indices, insert_summaries = self._get_insert_info(tup_list)
        if not insert_indices:
            return []

        for insert_index, summary in zip(insert_indices, insert_summaries):
            summary_tokens = self.processor.tokenize(summary)
            sample = sample.insert_field_after_index(
                'x', insert_index, summary_tokens)
        trans_samples = [sample]
        return trans_samples

    def _get_insert_info(self, tup_list):
        r"""
        Returns the index to insert the summary and the corresponding name.

        :param list tup_list: A list including dicts with sub sentence of
            original sentence and corresponding indices
        :return list indices: indices that will be insert
        :return list summaries: summaries that will be insert

        """
        insert_summaries = []
        insert_indices = []

        for item in tup_list:
            current_str = item['string']
            current_index = item['indices'][1]

            if current_str in self.entity_dict and current_index not in \
                    insert_indices:
                insert_indices.append(current_index-1)
                insert_summaries.append("(%s)" % self.entity_dict[current_str])
                continue

        if insert_indices:
            insert_indices, insert_summaries = zip(
                *sorted(zip(insert_indices, insert_summaries), reverse=True))

        return insert_indices, insert_summaries
