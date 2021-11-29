r"""
Add a distractor sentence to penalize MRC model
==========================================================
This transformation is based on CoreNLP, which is written in Java;
recent releases require Java 1.8+.
You need to have Java installed to run CoreNLP.
"""
import collections

from ..transformation import Transformation
from ....common.utils import FlintError
from ....input.component.sample.mrc_sample import MRCSample, \
    ConstituencyParse

__all__ = ['AddSentDiverse']


class AddSentDiverse(Transformation):
    r"""
    Generate a distractor before the sentence with answer.

    Example::

        origin question: Which NFL team represented the AFC at Super Bowl 50?
        transform distarctor: The UNICEF team of Kew Gardens represented
            the UNICEF at Champ Bowl 40.

    """
    def __init__(self):
        super().__init__()
        self.rules = collections.OrderedDict([
            # special tokens transformation
            ('special', MRCSample.alter_special),
            # synonym words in wordnet
            ('wn_antonyms', MRCSample.alter_wordnet_antonyms),
            ('nearbyNum', MRCSample.alter_nearby(
                ['CD'], ignore_pos=True)),                  # num
            ('nearbyProperNoun', MRCSample.alter_nearby(
                ['NNP', 'NNPS'])),                   # proper nouns
            ('nearbyProperNoun', MRCSample.alter_nearby(
                ['NNP', 'NNPS'], ignore_pos=True)),
            ('nearbyEntityNouns', MRCSample.alter_nearby(
                ['NN', 'NNS'], is_ner=True)),       # entity nouns
            ('nearbyEntityJJ', MRCSample.alter_nearby(
                ['JJ', 'JJR', 'JJS'], is_ner=True)),   # entity type
            ('entityType', MRCSample.alter_entity_type),
        ])

    def __repr__(self):
        return 'AddSentenceDiverse'

    def _transform(
            self,
            sample,
            nearby_word_dict=None,
            pos_tag_dict=None,
            **kwargs
    ):
        r"""
        Transform the question based on specific rules, replace the ground truth
        with fake answer, and then convert the question
        and fake answer to a distractor.

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
        question = sample.get_value('question')
        answers = sample.get_answers()
        answer_token_start = answers[0]['start']
        answer_text = answers[0]['text']
        sentences = sample.get_sentences('context')

        try:  # constituency parsing and linguistic feature generation
            question_tokens = self.processor.feature_extract(question)
            parse = self.processor.get_parser(question)
        except IOError:
            raise FlintError("Corenlp HTTPError, skip this sample")
        # Transform a sentence with AlterSentence Transformation
        alter_question, tokens, _ = sample.alter_sentence(
            question_tokens,
            nearby_word_dict=nearby_word_dict,
            pos_tag_dict=pos_tag_dict,
            rules=self.rules
        )

        assert len(tokens) == len(question_tokens)
        try:  # TODO
            const_parse = sample.read_const_parse(parse)
            const_parse = ConstituencyParse.replace_words(
                const_parse, [t['word'] for t in tokens])
        except IndexError:
            raise FlintError("Corenlp parsing mismatches spacy tokenizer")

        length = 0
        # Insert
        for i, sent in enumerate(sentences):

            if length + len(self.processor.tokenize(sent)) \
                    < answer_token_start:
                length = length + len(self.processor.tokenize(sent))
                continue
            sent_tokens = self.processor.feature_extract(sent)
            new_ans = sample.convert_answer(
                answer_text, sent_tokens, alter_question)
            distractor = sample.run_conversion(
                alter_question, new_ans, tokens, const_parse)
            if distractor and new_ans:
                # Insert the distract sentence before the answer
                new_sample = sample.insert_field_before_index(
                    'context', length, self.processor.tokenize(distractor))
                return [new_sample]
            else:
                return []
