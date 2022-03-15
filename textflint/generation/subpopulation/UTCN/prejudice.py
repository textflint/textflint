r"""
Extract samples with gender bias
============================================

"""
__all__ = ['PrejudiceSubPopulation']

from flashtext import KeywordProcessor
from ..subpopulation import SubPopulation
from ....common.settings import PREJUDICE_PATH
from ....common.utils.install import download_if_needed
from ....common.utils.file_io import read_json

MAN_WORDS = [
    "he",
    'him',
    'boy',
    'boy\'s',
    'man',
    'men',
    'his',
    'boys',
    'mr',
    'gentle',
    'husband'
]
WOMAN_WORDS = [
    'she',
    'her',
    'girl',
    'girl\'s',
    'woman',
    'women',
    'her',
    'girls',
    'mrs',
    'lady',
    'wife'
]


class PrejudiceSubPopulation(SubPopulation):
    r"""
    Filter samples based on gender bias

    for example in mode 'man'::

        sample 1: "There is a boy.", score: 1
        sample 2: "There is a girl.", score: 1
        sample 3: "There are boys and girls.", score: 0
    """
    def __init__(
        self,
        mode='man'
    ):
        super().__init__()

        self.mode = mode
        assert mode in ['man', 'woman', 'both', 'none'], \
            "Mode should be one in ['man', 'woman', 'both', 'none']"

        man_phrases, woman_phrases = self.get_data(
            download_if_needed(PREJUDICE_PATH))
        man_phrases.extend(self.get_words(MAN_WORDS))
        woman_phrases.extend(self.get_words(WOMAN_WORDS))
        self.processor = KeywordProcessor(case_sensitive=True)
        self.processor.add_keywords_from_dict({"man": man_phrases})
        self.processor.add_keywords_from_dict({"woman": woman_phrases})
        # TODO
        self.processor.remove_keyword('My')

    def __repr__(self):
        return "PrejudiceSubpopulation" + "-" + self.mode

    @staticmethod
    def get_data(path):
        # get the name dictionary

        for dic in read_json(path):
            _, dic = dic
            return dic['men'], dic['women']

    @staticmethod
    def get_words(words):
        tokens = []
        tokens.extend(words)
        tokens.extend([token.upper() for token in words])
        tokens.extend([token.title() for token in words])
        return tokens

    def word_match(self, texts, type):
        for text in texts:
            if type == 'man':
                result = self.processor.extract_keywords(text)
                if 'man' in result:
                    return True
            else:
                result = self.processor.extract_keywords(text)
                if 'woman' in result:
                    return True
        return False

    def _score(self, sample, fields, **kwargs):
        r"""
        1 or 0 indicate whether sample fields match mode and prejudice words

        :param sample: data sample
        :param list fields: list of field str
        :param kwargs:
        :return int: score for sample
        """

        texts = [sample.get_text(field) for field in fields]
        man_match = self.word_match(texts, type='man')
        woman_match = self.word_match(texts, type='woman')

        if self.mode == 'man':
            return man_match and not woman_match
        elif self.mode == 'woman':
            return woman_match and not man_match
        elif self.mode == 'both':
            return woman_match and man_match
        else:
            return not woman_match and not man_match

    def get_slice(self, scores, dataset):
        r"""
        Save the samples that mach the phrase groups and mode

        """
        sub_samples = []
        for i, sample in enumerate(dataset):
            if scores[i]:
                sub_samples.append(sample)
        return sub_samples
