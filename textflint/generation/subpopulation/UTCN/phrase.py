r"""
Extract samples matching phrases
============================================

"""
__all__ = ['PhraseSubPopulation']
from flashtext import KeywordProcessor
from ..subpopulation import SubPopulation
from ....common.settings import NEGATION, QUESTION


class PhraseSubPopulation(SubPopulation):
    r"""
    Filter samples based on a group of phrases

    for example, with phrase_name = 'question'::

        sample 1: "Who is Jack.", score: 1
        sample 2: "I am Jack.", score: 0
    """
    def __init__(
        self,
        phrase_name='negation'
    ):
        super().__init__()
        self.phrase_name = phrase_name
        if self.phrase_name == 'negation':
            self.phrases = NEGATION
        elif self.phrase_name == 'question':
            self.phrases = QUESTION

        self.phrase_processor = KeywordProcessor(case_sensitive=True)
        self.phrase_processor.add_keywords_from_dict(
            {self.phrase_name: self.phrases})

    def __repr__(self):
        return "PhraseSubPopulation" + "-" + self.phrase_name

    def phrase_match(self, text):
        match = False
        # Search for phrases
        result = self.phrase_processor.extract_keywords(text)
        if result:
            match = True
        return match

    def _score(self, sample, fields, **kwargs):
        """
        1 or 0 indicates whether sample fields match phrase groups

        :param sample: data sample
        :param list fields: list of field str
        :param kwargs:
        :return int: score for sample

        """

        text = ' '.join([sample.get_text(field) for field in fields])
        match = self.phrase_match(text)

        return match

    def get_slice(self, scores, dataset):
        r"""
        Save the samples that mach the phrase groups

        """
        sub_samples = []
        for i, sample in enumerate(dataset):
            if scores[i]:
                sub_samples.append(sample)
        return sub_samples
