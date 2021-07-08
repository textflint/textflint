r"""
Perturb Answer by altering the sentence that contains answer
==========================================================
"""

import collections

from ....common.settings import ORIGIN
from ..transformation import Transformation
from ....input.component.sample.mrc_sample import MRCSample


class PerturbAnswer(Transformation):
    r"""
    Transform the sentence containing answer with AlterSentence transformation.

    Example::

        origin sentence: Denver Broncos defeated the National Football
            Conference champion Carolina Panthers 24–10 to earn their
            third Super Bowl title.
        transformed sentence: Denver Broncos defeated the National Football
            Conference champ Carolina Panthers 24–10 to earn
            their 3rd Super Bowl rubric.
    """

    def __init__(self):
        super().__init__()

        # Rules for altering sentences
        self.rules = collections.OrderedDict([
            ('wn_synonyms', MRCSample.alter_wordnet_synonyms),
            ('nearbyProperNoun', MRCSample.alter_nearby(['NNP', 'NNPS'])),
            ('nearbyProperNoun', MRCSample.alter_nearby(
                ['NNP', 'NNPS'], ignore_pos=True)),
            ('nearbyEntityNouns', MRCSample.alter_nearby(
                ['NN', 'NNS'], is_ner=True)),
            ('nearbyEntityJJ', MRCSample.alter_nearby(
                ['JJ', 'JJR', 'JJS'], is_ner=True)),
        ])

    def __repr__(self):
        return 'PerturbAnswer'

    def _transform(
            self,
            sample,
            nearby_word_dict=None,
            pos_tag_dict=None,
            **kwargs
    ):
        r"""
        Extract the sentence with answer from context, replace synonyms
            based on WordNet and glove
        embedding space while keep the semantic meaning unchanged.
        :param sample: the sample to transform
        :param dict nearby_word_dict: the dict to search for nearby words
        :param dict pos_tag_dict: the dict to search for
            the most frequent pos tags
        :param kwargs:
        :return: list of sample
        """

        # filter no-answer samples
        if sample.is_impossible:
            return []
        answers = sample.get_answers()
        answer_token_start = answers[0]['start']
        answer_text = answers[0]['text']
        sentences = sample.get_sentences('context')

        sent_start = 0
        alter_sent = None
        indices = None
        # Pick up the sentence that contains the answer
        for i, sent in enumerate(sentences):
            if sent_start + len(self.processor.tokenize(sent)) \
                    <= answer_token_start:
                sent_start += len(self.processor.tokenize(sent))
                continue
            # deal with sentence tokenize error
            if sent.find(answer_text) < 0:
                return []
            sent = self.processor.feature_extract(sent)
            # Transform a sentence with AlterSentence function
            alter_sent, _, indices = sample.alter_sentence(
                sent, nearby_word_dict=nearby_word_dict,
                pos_tag_dict=pos_tag_dict, rules=self.rules)
            indices = [index + sent_start for index in indices]
            break
        if alter_sent is None:
            return None
        transform_samples = []
        results = []
        replace_items = []
        words = self.processor.tokenize(alter_sent)
        context_mask = sample.get_mask('context')

        for index in indices:
            if index >= len(context_mask):
                return []
            if context_mask[index] != ORIGIN:
                continue
            results.append(index)
            replace_items.append(words[index - sent_start])
        if results:
            new_sample = sample.replace_field_at_indices(
                'context', results, replace_items)
        else:
            return []
        transform_samples.append(new_sample)

        return transform_samples
