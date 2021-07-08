r"""
Add the difference part of target in ABSA task
==========================================================
"""

import random
import string
from .absa_transformation import ABSATransformation

__all__ = ['AddDiff']


class AddDiff(ABSATransformation):
    r"""
    Add the difference part of aspect to the end of original sentence.
    The difference part is extracted from the training set of SemEval2014
    or user's customization.

    Example::

        Original sentence: "BEST spicy tuna roll, great asian salad.
        （Target: spicy tuna roll)"
        Transformed sentence: "BEST spicy tuna roll, great asian salad,
        but this small place is packed,
        on a cold day, the seating by the entrance way can be pretty drafty and
        bad service."

    """

    def __init__(
            self,
            language="eng"):
        super().__init__()

        if language != "eng":
            raise ValueError(f"Language {language} is not available.")
        self.language = language
        self.tokenize = self.processor.tokenize
        self.untokenize = self.processor.inverse_tokenize

    def __repr__(self):
        return "AddDiff"

    def _transform(self, sample, n=1, field='sentence',
                   extra_text=None, **kwargs):
        r"""
        Transform data sample to a list of Sample.

        :param ~textflint.input.component.sample.ABSAsample sample: input ABSAsample
        :param int n: the number of transformation, in ABSA-specific transformations n=1
        :param str field:field name
        :param dict extra_text: extra text will be added to the original sentence
        :return list: list of transformed ABSAsample

        """
        trans_samples = []
        self.trans_term_list = sample.term_list
        self.terms = sample.terms
        self.term_list = sample.term_list
        self.sentence = sample.sentence.text
        if sample.trans_id is None:
            self.trans_id = [idx for idx in self.terms]
        else:
            self.trans_id = [sample.trans_id]
        all_term = [self.term_list[idx]['term'] for idx in self.term_list]

        for term_id in self.trans_id:
            trans_sample = sample.clone(sample)
            polarity = self.term_list[term_id]['polarity']
            if polarity == 'positive':
                add_text = extra_text['negative']
            elif polarity == 'negative':
                add_text = extra_text['positive']
            else:
                add_text = extra_text['neutral']

            add_sentence = self._get_add_sentence(add_text, all_term)
            trans_sentence = self._concatenate_sentence(add_sentence)
            trans_sample.update_sentence(trans_sentence)
            trans_sample.trans_id = term_id
            trans_samples.append(trans_sample)

        return trans_samples

    def _get_add_sentence(self, add_text, all_term):
        r"""
        Get the sentence that owns different polarity compared with
        the aspect. Choose 1~3 sentences randomly from add_text and put
        them together.

        :param list add_text: extra text
        :param list all_term: all aspect term
        :return list: extra sentence that need to be added to original sentence

        """
        sentence = self.sentence
        punctuation = '.'
        if sentence[-1] == string.punctuation:
            punctuation = sentence[-1]

        while True:
            add_num = random.randint(1, 3)
            rand_num = random.sample(range(len(add_text)), 3)
            rand_terms = [add_text[i][0] for i in rand_num]
            rand_sentences = [self.tokenize(add_text[i][1][0])
                              for i in rand_num]

            for idx, ran_sentence in enumerate(rand_sentences):
                if rand_terms[idx] in all_term:
                    break
                if ran_sentence[-1] in string.punctuation:
                    rand_sentences[idx] = ran_sentence[:-1]
            if add_num == 3:
                add_sentence = rand_sentences[0] + [','] + rand_sentences[1] + [
                    'and'] + rand_sentences[2] + [punctuation]
            elif add_num == 2:
                add_sentence = rand_sentences[0] + ['and'] + rand_sentences[1] \
                               + [punctuation]
            else:
                add_sentence = rand_sentences[0] + [punctuation]
            break

        return add_sentence

    def _concatenate_sentence(self, add_sentence):
        r"""
        Concatenate the extra part to original sentence.

        :param list add_sentence: extra sentence that need to be
            added to original sentence
        :return list: transformed sentence

        """
        sentence = self.sentence[:-1]
        opi_tag = self.get_postag(add_sentence, 0, 1)
        if opi_tag[0] != 'CONJ':
            trans_sentence = self.untokenize(add_sentence)
            if 'but' in sentence or 'although' in sentence:
                trans_sentence = sentence + "; " + trans_sentence
            else:
                trans_sentence = sentence + ", but " + trans_sentence
        else:
            trans_sentence = self.untokenize(add_sentence)
            trans_sentence = sentence + ". " + trans_sentence[
                0].upper() + trans_sentence[1:]
        return trans_sentence
