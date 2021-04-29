r"""
MRC Sample Class
==========================================================
Manage text transformation for MRC.
Heavily borrowed from adversarial-squad.
For code in adversarial-squad, please check the following link:
https://github.com/robinjia/adversarial-squad
"""

from .sample import Sample
from ..field import Field, TextField
from ....common.utils.list_op import normalize_scope
from ....common.preprocess.nltk_res_load import ModelManager
from ....common.settings import *

from nltk.stem.lancaster import LancasterStemmer
from copy import deepcopy


__all__ = ['MRCSample']


class ConstituencyParse(object):
    """A CoreNLP constituency parse (or a node in a parse tree).

    Word-level constituents have |word| and |index| set and no children.
    Phrase-level constituents have no |word| or |index|
    and have at least one child.

    """

    def __init__(self, tag, children=None, word=None, index=None):
        self.tag = tag
        if children:
            self.children = children
        else:
            self.children = None
        self.word = word
        self.index = index

    @classmethod
    def _recursive_parse_corenlp(cls, tokens, i, j):

        if tokens[i] == '(':
            tag = tokens[i + 1]
            children = []
            i = i + 2
            while True:
                child, i, j = cls._recursive_parse_corenlp(tokens, i, j)
                if isinstance(child, cls):
                    children.append(child)
                    if tokens[i] == ')':
                        return cls(tag, children), i + 1, j
                else:
                    if tokens[i] != ')':
                        raise ValueError('Expected ")" following leaf')
                    return cls(tag, word=child, index=j), i + 1, j + 1
        else:
            # Only other possibility is it's a word
            return tokens[i], i + 1, j

    @classmethod
    def from_corenlp(cls, s):
        r"""
        Parses the "parse" attribute returned by CoreNLP parse annotator.

        """
        s_spaced = s.replace('\n', ' ').replace('(', ' ( ').replace(')', ' ) ')
        tokens = [t for t in s_spaced.split(' ') if t]
        tree, index, num_words = cls._recursive_parse_corenlp(tokens, 0, 0)
        if index != len(tokens):
            raise ValueError(
                'Only parsed %d of %d tokens' %
                (index, len(tokens)))
        return tree

    def is_singleton(self):
        if self.word:
            return True
        if len(self.children) > 1:
            return False
        return self.children[0].is_singleton()

    def get_phrase(self):
        if self.word:
            return self.word
        toks = []
        for i, c in enumerate(self.children):
            p = c.get_phrase()
            if i == 0 or p.startswith("'"):
                toks.append(p)
            else:
                toks.append(' ' + p)
        return ''.join(toks)

    def get_start_index(self):
        if self.index is not None:
            return self.index
        return self.children[0].get_start_index()

    def get_end_index(self):
        if self.index is not None:
            return self.index + 1
        return self.children[-1].get_end_index()

    @classmethod
    def _recursive_replace_words(cls, tree, new_words, i):
        if tree.word:
            new_word = new_words[i]
            return cls(tree.tag, word=new_word, index=tree.index), i + 1
        new_children = []
        for c in tree.children:
            new_child, i = cls._recursive_replace_words(c, new_words, i)
            new_children.append(new_child)
        return cls(tree.tag, children=new_children), i

    @classmethod
    def replace_words(cls, tree, new_words):
        """Return a new tree, with new words replacing old ones."""
        new_tree, i = cls._recursive_replace_words(tree, new_words, 0)
        return new_tree


class MRCSample(Sample):
    r"""
    MRC Sample class to hold the mrc data info and provide atomic operations.

    """

    STEMMER = LancasterStemmer()
    model_manager = ModelManager()
    wn = model_manager.load(NLTK_WORDNET)
    POS_TO_WORDNET = {
        'NN': wn.NOUN,
        'JJ': wn.ADJ,
        'JJR': wn.ADJ,
        'JJS': wn.ADJ,
    }

    def __init__(
        self,
        data,
        origin=None,
        sample_id=None
    ):
        r"""
        The sample object for machine reading comprehension task
        :param dict data: The dict obj that contains data info.
        :param bool origin:
        :param int sample_id: sample index

        """
        self.context = None
        self.question = None
        self.title = None
        self.is_impossible = None
        self.answers = None
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'MRCSample'

    def check_data(self, data):
        r"""
        Check whether the input data is legal
        :param dict data: dict obj that contains data info

        """
        assert 'context' in data and isinstance(data['context'], str), \
            "Context should be in data, and the type of context should be str"
        assert 'question' in data and isinstance(data['question'], str), \
            "Question should be in data, and the type of question should be str"
        assert 'answers' in data and isinstance(data['answers'], list), \
            "Answers should be in data, and the type of answers should be dict"
        assert 'title' in data, "Title should be in data."
        assert 'is_impossible' in data, "Is_possible should be in data"

    def is_legal(self):
        r"""
        Validate whether the sample is legal
        :return: bool

        """
        import re

        def normalize_answer(s):
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)

            def white_space_fix(text):
                return ' '.join(text.split())

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(lower(s)))

        for answer in self.answers:
            answer_text = self.text_processor.inverse_tokenize(
                self.context.words[answer['start']:answer['end'] + 1]
            )
            if normalize_answer(answer_text) != \
                    normalize_answer(answer['text']):
                return False
            answer['text'] = answer_text
        return True

    @staticmethod
    def convert_idx(text, tokens):
        r"""
        Get the start and end character idx of tokens in the context

        :param str text: context text
        :param list tokens: context words
        :return: list of spans

        """
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                print("Token {} cannot be found".format(token))
            spans.append((current, current + len(token)))
            current += len(token)
        return spans

    def load_answers(self, ans, spans):
        r"""
        Get word-level positions of answers

        :param dict ans: answers dict with character position and text
        :param list spans: the start idx and end idx of tokens

        """
        answers = [answer['text'] for answer in ans]
        span_starts = [answer['answer_start']
                       for answer in ans]
        span_ends = [start + len(answer)
                     for start, answer in zip(span_starts, answers)]
        for answer, sos, eos in zip(answers, span_starts, span_ends):
            y1, y2 = self.get_answer_position(spans, sos, eos)
            for i in range(y1, y2 + 1):
                self.context.set_mask(i, 1)
            self.answers.append({
                'text': answer,
                'start': y1,
                'end': y2
            })

    def get_answers(self):
        r"""
        Get copy of answers

        :return: dict, answers

        """
        return deepcopy(self.answers)

    def set_answers_mask(self):
        r"""
        Set the answers with TASK_MASK

        """
        for answer in self.answers:
            y1, y2 = answer['start'], answer['end']
            if y2 > len(self.get_words('context')) - 1 or y1 < 0:
                raise ValueError
            for i in range(y1, y2 + 1):
                self.context.set_mask(i, 1)

    def load(self, data):
        r"""
        Convert data dict which contains essential information to MRCSample.

        :param dict data: the dict obj that contains dict info

        """
        self.context = TextField(data['context'])
        self.question = TextField(data['question'])
        self.title = Field(data['title'])
        if isinstance(data['is_impossible'], bool):
            self.is_impossible = data['is_impossible']
        elif isinstance(data['is_impossible'], str):
            self.is_impossible = True if data['is_impossible'] == 'True' \
                else False
        self.is_impossible = data['is_impossible']
        self.answers = []

        if self.is_impossible:
            self.answers = []
        else:
            spans = self.convert_idx(data['context'], self.context.words)
            self.load_answers(data['answers'], spans)

            if not self.is_legal():
                raise ValueError("Data sample {0} is not legal, "
                                 "Answer spans mismatch answer text."
                                 .format(data))

    def dump(self):
        r"""
        Convert data dict which contains essential information to MRCSample.

        :return: dict object

        """

        if self.is_impossible:
            answers = []
        else:
            if not self.is_legal():
                raise ValueError("Answer spans mismatch answer text.")
            spans = self.convert_idx(self.context.text, self.context.words)
            answers = [{'text': answer['text'],
                        'answer_start': spans[answer['start']][0]}
                       for answer in self.answers]
        return {
            'context': self.context.text,
            'question': self.question.text,
            'answers': answers,
            'title': self.title.field_value,
            'is_impossible': self.is_impossible,
            "sample_id": self.sample_id}

    def delete_field_at_index(self, field, index):
        r"""
        Delete the word seat in del_index.

        :param str field:field name
        :param int|list|slice index: modified scope
        :return: modified sample

        """
        return self.delete_field_at_indices(field, [index])

    def delete_field_at_indices(self, field, indices):
        r"""
        Delete items of given scopes of field value.

        :param str field: field name
        :param list indices: list of int/list/slice, modified scopes
        :return: modified Sample

        """
        answers = deepcopy(self.answers)
        for index in indices:
            scope = normalize_scope(index)
            offset = scope[1] - scope[0]
            for answer in answers:
                if scope[1] < answer['start']:
                    answer['start'] -= offset
                    answer['end'] -= offset
        sample = super(MRCSample, self).delete_field_at_indices(field, indices)
        sample.answers = answers
        return sample

    def insert_field_before_indices(self, field, indices, items):
        r"""
        Insert items of multi given scopes before indices of
        field value at the same time.

        :param str field: field name
        :param list indices: list of int/list/slice, modified scopes
        :param list items: inserted items
        :return: modified Sample

        """
        answers = deepcopy(self.answers)
        for i, index in enumerate(indices):
            if isinstance(items[i], list):
                offset = len(items[i])
            else:
                items[i] = MRCSample.text_processor.tokenize(items[i])
                offset = len(items[i])

            for answer in answers:
                if index <= answer['start']:
                    answer['start'] += offset
                    answer['end'] += offset
        sample = super(MRCSample, self).insert_field_before_indices(
            field, indices, items)
        sample.answers = answers
        return sample

    def get_answer_rules(self):
        answer_rules = [
            ('ner_person', self.ans_entity_full('PERSON', 'Jeff Dean')),
            ('ner_location', self.ans_entity_full('LOCATION', 'Chicago')),
            ('ner_organization', self.ans_entity_full(
                'ORGANIZATION', 'Stark Industries')),
            ('ner_misc', self.ans_entity_full('MISC', 'Jupiter')),
            ('abbrev', self.ans_abbrev('LSTM')),
            ('wh_who', self.ans_match_wh('who', 'Jeff Dean')),
            ('wh_when', self.ans_match_wh('when', '1956')),
            ('wh_where', self.ans_match_wh('where', 'Chicago')),
            ('wh_where', self.ans_match_wh('how many', '42')),
            # Starts with verb
            ('pos_begin_vb', self.ans_pos('VB', 'learn')),
            ('pos_end_vbd', self.ans_pos('VBD', 'learned')),
            ('pos_end_vbg', self.ans_pos('VBG', 'learning')),
            ('pos_end_vbp', self.ans_pos('VBP', 'learns')),
            ('pos_end_vbz', self.ans_pos('VBZ', 'learns')),
            # Ends with some POS tag
            ('pos_end_nn',
             self.ans_pos('NN', 'hamster', end=True, add_dt=True)),
            ('pos_end_nnp', self.ans_pos('NNP', 'Central Park',
                                              end=True, add_dt=True)),
            ('pos_end_nns', self.ans_pos('NNS', 'hamsters',
                                              end=True, add_dt=True)),
            ('pos_end_nnps', self.ans_pos('NNPS', 'Kew Gardens',
                                               end=True, add_dt=True)),
            ('pos_end_jj', self.ans_pos('JJ', 'deep', end=True)),
            ('pos_end_jjr', self.ans_pos('JJR', 'deeper', end=True)),
            ('pos_end_jjs', self.ans_pos('JJS', 'deepest', end=True)),
            ('pos_end_rb', self.ans_pos('RB', 'silently', end=True)),
            ('pos_end_vbg', self.ans_pos('VBG', 'learning', end=True)),
            ('catch_all', self.ans_catch_all('aliens')),
        ]
        return answer_rules

    def insert_field_before_index(self, field, index, items):
        r"""
        Insert item before index of field value.

        :param str field: field name
        :param int index: modified scope
        :param items: inserted item
        :return: modified Sample

        """
        return self.insert_field_before_indices(field, [index], [items])

    def insert_field_after_index(self, field, index, new_item):
        r"""
        Insert item after index of field value.

        :param str field: field name
        :param int index: modified scope
        :param new_item: inserted item
        :return: modified Sample

        """

        return self.insert_field_after_indices(field, [index], [new_item])

    def insert_field_after_indices(self, field, indices, items):
        r"""
        Insert items of multi given scopes after indices of
        field value at the same time.

        :param str field: field name
        :param list indices: list of int/list/slice, modified scopes
        :param list items: inserted items
        :return: modified Sample

        """
        answers = deepcopy(self.answers)
        for i, index in enumerate(indices):
            if isinstance(items[i], list):
                offset = len(items[i])
            else:
                items[i] = MRCSample.text_processor.tokenize(items[i])
                offset = len(items[i])

            for answer in answers:
                if index < answer['start']:
                    answer['start'] += offset
                    answer['end'] += offset
        sample = super(MRCSample, self).insert_field_after_indices(
            field, indices, items)
        sample.answers = answers
        return sample

    def unequal_replace_field_at_indices(self, field, indices, rep_items):
        r"""
        Replace scope items of field value with rep_items
        which may not equal with scope.

        :param str field: field name
        :param list indices: list of int/list/slice, modified scopes
        :param list rep_items: replace items
        :return: modified sample

        """
        assert len(indices) == len(rep_items) > 0
        sample = self.clone(self)
        sorted_items, sorted_indices = zip(
            *sorted(zip(rep_items, indices), key=lambda x: x[1], reverse=True))

        for idx, sorted_token in enumerate(sorted_items):
            sample = sample.delete_field_at_index(field, sorted_indices[idx])
            insert_index = sorted_indices[idx] if isinstance(
                sorted_indices[idx], int) else sorted_indices[idx][0]
            field_obj = getattr(sample, field)
            if insert_index > len(field_obj):
                raise ValueError(
                    'Cant replace items at range {0}'.format(
                        sorted_indices[idx]))
            elif insert_index == len(field_obj):
                sample = sample.insert_field_after_index(
                    field, insert_index - 1, sorted_token)
            else:
                sample = sample.insert_field_before_index(
                    field, insert_index, sorted_token)

        return sample

    @staticmethod
    def get_answer_position(spans, answer_start, answer_end):
        r"""
        Get answer tokens start position and end position

        """
        answer_span = []
        for idx, span in enumerate(spans):
            if not (answer_end <= span[0] or answer_start >= span[1]):
                answer_span.append(idx)
        assert len(answer_span) > 0
        y1, y2 = answer_span[0], answer_span[-1]
        return y1, y2

    @staticmethod
    def run_conversion(question, answer, tokens, const_parse):
        """
        Convert the question and answer to a declarative sentence

        :param str question: question
        :param str answer: answer
        :param list tokens: the semantic tag dicts of question
        :param const_parse: the constituency parse of question
        :return: a declarative sentence

        """

        for rule in CONVERSION_RULES:
            sent = rule.convert(question, answer, tokens, const_parse)
            if sent:
                return sent
        return None

    def convert_answer(self, answer, sent_tokens, question):
        """
        Replace the ground truth with fake answer based on specific rules

        :param str answer: ground truth, str
        :param list sent_tokens: sentence dicts, like [{'word': 'Saint',
            'pos': 'NNP', 'lemma': 'Saint', 'ner': 'PERSON'}...]
        :param str question: question sentence
        :return str: fake answer

        """
        tokens = MRCSample.get_answer_tokens(sent_tokens, answer)
        determiner = MRCSample.get_determiner_for_answers(answer)
        for rule_name, func in self.get_answer_rules():
            new_ans = func(answer, tokens, question, determiner=determiner)
            if new_ans:
                return new_ans
        return None

    @staticmethod
    def alter_sentence(
        sample,
        nearby_word_dict=None,
        pos_tag_dict=None,
        rules=None
    ):
        """

        :param sample: sentence dicts, like [{'word': 'Saint', 'pos': 'NNP',
            'lemma': 'Saint', 'ner': 'PERSON'}...]
        :param nearby_word_dict: the dictionary to search for nearby words
        :param pos_tag_dict: the dictionary to search for
            the most frequent pos tags
        :param rules: the rules to alter the sentence
        :return: alter_sentence, alter_sentence dicts

        """
        used_words = [t['word'].lower() for t in sample]
        indices = []
        sentence = []
        new_sample = []
        for i, t in enumerate(sample):
            if t['word'].lower() in DO_NOT_ALTER:
                sentence.append(t['word'])
                new_sample.append(t)
                continue
            found = False
            for rule_name in rules:
                rule = rules[rule_name]
                new_words = rule(t, nearby_word_dict=nearby_word_dict,
                                 pos_tag_dict=pos_tag_dict)
                if new_words:
                    for nw in new_words:
                        if nw.lower() in used_words:
                            continue
                        if nw.lower() in BAD_ALTERATIONS:
                            continue
                        # Match capitalization
                        if t['word'] == t['word'].upper():
                            nw = nw.upper()
                        elif t['word'] == t['word'].title():
                            nw = nw.title()
                        found = True
                        sentence.append(nw)
                        new_sample.append({'word': nw,
                                           'lemma': nw,
                                           'pos': t['pos'],
                                           'ner': t['ner']
                                           })
                        indices.append(i)
                        break
                if found:
                    break
            if not found:
                sentence.append(t['word'])
                new_sample.append(t)

        return " ".join(sentence), new_sample, indices

    # TODO, remove kwargs
    @staticmethod
    def alter_special(token, **kwargs):
        """
        Alter special tokens

        :param token: the token to alter
        :param kwargs:
        :return: like 'US' ->  'UK'

        """
        w = token['word']
        if w in SPECIAL_ALTERATIONS:
            return [SPECIAL_ALTERATIONS[w]]
        return None

    @staticmethod
    def alter_wordnet_antonyms(token, **kwargs):
        """
        Replace words with wordnet antonyms

        :param token: the token to replace
        :param kwargs:
        :return: like good -> bad

        """
        if token['pos'] not in MRCSample.POS_TO_WORDNET:
            return None
        w = token['word'].lower()
        wn_pos = MRCSample.POS_TO_WORDNET[token['pos']]
        synsets = MRCSample.wn.synsets(w, wn_pos)
        if not synsets:
            return None
        synset = synsets[0]
        antonyms = []

        for lem in synset.lemmas():
            if lem.antonyms():
                for a in lem.antonyms():
                    new_word = a.name()
                    if '_' in a.name():
                        continue
                    antonyms.append(new_word)
        return antonyms

    @staticmethod
    def alter_wordnet_synonyms(token, **kwargs):
        """
        Replace words with synonyms

        :param token: the token to replace
        :param kwargs:
        :return: like good -> great

        """

        if token['pos'] not in MRCSample.POS_TO_WORDNET:
            return None
        w = token['word'].lower()
        wn_pos = MRCSample.POS_TO_WORDNET[token['pos']]
        synsets = MRCSample.wn.synsets(w, wn_pos)
        if not synsets:
            return None
        synonyms = []

        for syn in synsets:
            for syn_word in syn.lemma_names():
                if (
                        (syn_word != w)
                        and ("_" not in syn_word)
                ):
                    # WordNet can suggest phrases that are joined by '_' but we
                    # ignore phrases.
                    synonyms.append(syn_word)
        return synonyms

    @staticmethod
    def alter_nearby(pos_list, ignore_pos=False, is_ner=False):
        """
        Alter words based on glove embedding space

        :param pos_list: pos tags list
        :param bool ignore_pos: whether to match pos tag
        :param bool is_ner: indicate ner
        :return: like 'Mary' -> 'Rose'

        """
        def func(token, nearby_word_dict=None, pos_tag_dict=None, **kwargs):
            if token['pos'] not in pos_list:
                return None
            if is_ner and token['ner'] not in (
                    'PERSON', 'LOCATION', 'ORGANIZATION', 'MISC'):
                return None
            w = token['word'].lower()
            if w in 'war':
                return None
            if w not in nearby_word_dict:
                return None
            new_words = []
            w_stem = MRCSample.STEMMER.stem(w.replace('.', ''))
            for x in nearby_word_dict[w][1:]:
                new_word = x['word']
                # Make sure words aren't too similar (e.g. same stem)
                new_stem = MRCSample.STEMMER.stem(new_word.replace('.', ''))
                if w_stem.startswith(new_stem) or new_stem.startswith(w_stem):
                    continue
                if not ignore_pos:
                    # Check for POS tag match
                    if new_word not in pos_tag_dict:
                        continue
                    new_postag = pos_tag_dict[new_word]
                    if new_postag != token['pos']:
                        continue
                new_words.append(new_word)
            return new_words

        return func

    @staticmethod
    def alter_entity_type(token, **kwargs):
        """
        Alter entity

        :param token: the word to replace
        :param kwargs:
        :return: like 'London' -> 'Berlin'

        """
        pos = token['pos']
        ner = token['ner']
        word = token['word']
        is_abbrev = (word == word.upper() and not word == word.lower())
        if token['pos'] not in (
            'JJ',
            'JJR',
            'JJS',
            'NN',
            'NNS',
            'NNP',
            'NNPS',
            'RB',
            'RBR',
            'RBS',
            'VB',
            'VBD',
            'VBG',
            'VBN',
            'VBP',
                'VBZ'):
            # Don't alter non-content words
            return None
        if ner == 'PERSON':
            return ['Jackson']
        elif ner == 'LOCATION':
            return ['Berlin']
        elif ner == 'ORGANIZATION':
            if is_abbrev:
                return ['UNICEF']
            return ['Acme']
        elif ner == 'MISC':
            return ['Neptune']
        elif ner == 'NNP':
            if is_abbrev:
                return ['XKCD']
            return ['Dalek']
        elif pos == 'NNPS':
            return ['Daleks']

        return None

    @staticmethod
    def get_determiner_for_answers(a):
        words = a.split(' ')
        if words[0].lower() == 'the':
            return 'the'
        if words[0].lower() in ('a', 'an'):
            return 'a'
        return None

    @staticmethod
    def get_answer_tokens(sent_tokens, answer):
        """
        Extract the pos, ner, lemma tags of answer tokens

        :param list sent_tokens: a list of dicts
        :param str answer: answer
        :return: a list of dicts
            like [
            {'word': 'Saint', 'pos': 'NNP', 'lemma': 'Saint', 'ner': 'PERSON'},
            {'word': 'Bernadette', 'pos': 'NNP', 'lemma': 'Bernadette', ...},
            {'word': 'Soubirous', 'pos': 'NNP', 'lemma': 'Soubirous', ...]
            ]

        """

        sent = " ".join([t['word'] for t in sent_tokens])
        start = sent.find(answer)
        end = start + len(answer)
        tokens = []
        length = 0
        for i, tok in enumerate(sent_tokens):
            if length > end:
                break
            if start <= length < end:
                tokens.append(tok)
            length = length + 1 + len(tok['word'])
        return tokens

    @staticmethod
    def ans_entity_full(ner_tag, new_ans):
        """
        Returns a function that yields new_ans iff every token has |ner_tag|

        :param str ner_tag: ner tag
        :param list new_ans: like [{'word': 'Saint', 'pos': 'NNP',
            'lemma': 'Saint', 'ner': 'PERSON'}...]
        :return: fake answer, str

        """
        def func(a, tokens, q, **kwargs):
            for t in tokens:
                if t['ner'] != ner_tag:
                    return None
            return new_ans

        return func

    @staticmethod
    def ans_abbrev(new_ans):
        """

        :param strnew_ans: answer words
        :return str: fake answer

        """
        def func(a, tokens, q, **kwargs):
            s = a
            if s == s.upper() and s != s.lower():
                return new_ans
            return None

        return func

    @staticmethod
    def ans_match_wh(wh_word, new_ans):
        """
        Returns a function that yields new_ans
            if the question starts with |wh_word|

        :param str wh_word: question word
        :param list new_ans: like [{'word': 'Saint', 'pos': 'NNP',
            'lemma': 'Saint', 'ner': 'PERSON'}...]
        :return str: fake answers,

        """
        def func(a, tokens, q, **kwargs):
            if q.lower().startswith(wh_word + ' '):
                return new_ans
            return None

        return func

    @staticmethod
    def ans_pos(pos, new_ans, end=False, add_dt=False):
        """
        Returns a function that yields new_ans if the first/last token has |pos|

        :param str pos: pos tag
        :param list new_ans: like [{'word': 'Saint', 'pos': 'NNP',
            'lemma': 'Saint', 'ner': 'PERSON'}...]
        :param bool end: whether to use the last word to match the pos tag
        :param bool add_dt: whether to add a determiner
        :return str: fake answer

        """
        def func(a, tokens, q, determiner, **kwargs):
            if end:
                t = tokens[-1]
            else:
                t = tokens[0]
            if t['pos'] != pos:
                return None
            if add_dt and determiner:
                return '%s %s' % (determiner, new_ans)
            return new_ans

        return func

    @staticmethod
    def ans_catch_all(new_ans):
        def func(a, tokens, q, **kwargs):
            return new_ans

        return func

    @staticmethod
    def compress_whnp(tree, inside_whnp=False):
        if not tree.children:
            return tree  # Reached leaf
        # Compress all children
        for i, c in enumerate(tree.children):
            tree.children[i] = MRCSample.compress_whnp(
                c, inside_whnp=inside_whnp or tree.tag == 'WHNP')

        if tree.tag != 'WHNP':
            if inside_whnp:
                # Wrap everything in an NP
                return ConstituencyParse('NP', children=[tree])
            return tree
        wh_word = None
        new_np_children = []
        new_siblings = []

        for i, c in enumerate(tree.children):
            if i == 0:
                if c.tag in ('WHNP', 'WHADJP', 'WHAVP', 'WHPP'):
                    wh_word = c.children[0]
                    new_np_children.extend(c.children[1:])
                elif c.tag in ('WDT', 'WP', 'WP$', 'WRB'):
                    wh_word = c
                else:
                    # No WH-word at start of WHNP
                    return tree
            else:
                if c.tag == 'SQ':  # Due to bad parse, SQ may show up here
                    new_siblings = tree.children[i:]
                    break
                # Wrap everything in an NP
                new_np_children.append(ConstituencyParse('NP', children=[c]))

        if new_np_children:
            new_np = ConstituencyParse('NP', children=new_np_children)
            new_tree = ConstituencyParse('WHNP', children=[wh_word, new_np])
        else:
            new_tree = tree
        if new_siblings:
            new_tree = ConstituencyParse(
                'SBARQ', children=[new_tree] + new_siblings)

        return new_tree

    @staticmethod
    def read_const_parse(parse_str):
        """
        Construct a constituency tree based on constituency parser

        """
        tree = ConstituencyParse.from_corenlp(parse_str)
        new_tree = MRCSample.compress_whnp(tree)
        return new_tree

    @staticmethod
    # Rules for converting questions into declarative sentences
    def fix_style(s):
        """
        Minor, general style fixes for questions.

        """
        s = s.replace('?', '')  # Delete question marks anywhere in sentence.
        s = s.strip(' .')
        if s[0] == s[0].lower():
            s = s[0].upper() + s[1:]
        return s + '.'

    @staticmethod
    def _check_match(node, pattern_tok):
        if pattern_tok in CONST_PARSE_MACROS:
            pattern_tok = CONST_PARSE_MACROS[pattern_tok]
        if ':' in pattern_tok:
            lhs, rhs = pattern_tok.split(':')
            match_lhs = MRCSample._check_match(node, lhs)
            if not match_lhs:
                return False
            phrase = node.get_phrase().lower()
            retval = any(phrase.startswith(w) for w in rhs.split('/'))
            return retval
        elif '/' in pattern_tok:
            return any(MRCSample._check_match(node, t)
                       for t in pattern_tok.split('/'))

        return ((pattern_tok.startswith('$') and pattern_tok[1:] == node.tag) or
                (node.word and pattern_tok.lower() == node.word.lower()))

    @staticmethod
    def _recursive_match_pattern(pattern_toks, stack, matches):
        """
        Recursively try to match a pattern, greedily.

        """
        if len(matches) == len(pattern_toks):
            # We matched everything in the pattern; also need stack to be empty
            return len(stack) == 0
        if len(stack) == 0:
            return False
        cur_tok = pattern_toks[len(matches)]
        node = stack.pop()
        # See if we match the current token at this level
        is_match = MRCSample._check_match(node, cur_tok)

        if is_match:
            cur_num_matches = len(matches)
            matches.append(node)
            new_stack = list(stack)
            success = MRCSample._recursive_match_pattern(
                pattern_toks, new_stack, matches)
            if success:
                return True
            # Backtrack
            while len(matches) > cur_num_matches:
                matches.pop()
        # Recurse to children
        if not node.children:
            return False  # No children to recurse on, we failed
        # Leftmost children should be popped first
        stack.extend(node.children[::-1])

        return MRCSample._recursive_match_pattern(pattern_toks, stack, matches)

    @staticmethod
    def match_pattern(pattern, const_parse):
        pattern_toks = pattern.split(' ')
        whole_phrase = const_parse.get_phrase()
        if whole_phrase.endswith('?') or whole_phrase.endswith('.'):
            # Match trailing punctuation as needed
            pattern_toks.append(whole_phrase[-1])
        matches = []
        success = MRCSample._recursive_match_pattern(
            pattern_toks, [const_parse], matches)
        if success:
            return matches
        else:
            return None

    # TODO
    @staticmethod
    def convert_whp(node, q, a, tokens):
        if node.tag in ('WHNP', 'WHADJP', 'WHADVP', 'WHPP'):
            # Apply WHP rules
            cur_phrase = node.get_phrase()
            cur_tokens = tokens[node.get_start_index():node.get_end_index()]
            for i, r in enumerate(WHP_RULES):
                phrase = r.convert(
                    cur_phrase, a, cur_tokens, node, run_fix_style=False)
                if phrase:
                    return phrase
        return None


class ConversionRule(object):
    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        raise NotImplementedError


class ConstituencyRule(ConversionRule):
    r"""
    A rule for converting question to sentence based on constituency parse.

    """

    def __init__(self, in_pattern, out_pattern, postproc=None):
        self.in_pattern = in_pattern  # e.g. "where did $NP $VP"
        self.out_pattern = out_pattern
        # e.g. "{1} did {2} at {0}."  Answer is always 0
        self.name = in_pattern
        if postproc:
            self.postproc = postproc
        else:
            self.postproc = {}

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        # Don't care about trailing punctuation
        pattern_toks = self.in_pattern.split(' ')
        match = MRCSample.match_pattern(self.in_pattern, const_parse)
        appended_clause = False

        if not match:
            # Try adding a PP at the beginning
            appended_clause = True
            new_pattern = '$PP , ' + self.in_pattern
            pattern_toks = new_pattern.split(' ')
            match = MRCSample.match_pattern(new_pattern, const_parse)
        if not match:
            # Try adding an SBAR at the beginning
            new_pattern = '$SBAR , ' + self.in_pattern
            pattern_toks = new_pattern.split(' ')
            match = MRCSample.match_pattern(new_pattern, const_parse)
        if not match:
            return None
        appended_clause_match = None
        fmt_args = [a]

        for t, m in zip(pattern_toks, match):
            if t.startswith('$') or '/' in t:
                # First check if it's a WHP
                phrase = MRCSample.convert_whp(m, q, a, tokens)
                if not phrase:
                    phrase = m.get_phrase()
                fmt_args.append(phrase)
        if appended_clause:
            appended_clause_match = fmt_args[1]
            fmt_args = [a] + fmt_args[2:]
        output = self.gen_output(fmt_args)
        if appended_clause:
            output = appended_clause_match + ', ' + output
        if run_fix_style:
            output = MRCSample.fix_style(output)

        return output

    def gen_output(self, fmt_args):
        """
        By default, use self.out_pattern.  Can be overridden.

        """
        return self.out_pattern.format(*fmt_args)


class ReplaceRule(ConversionRule):
    r"""
    A simple rule that replaces some tokens with the answer.

    """

    def __init__(self, target, replacement='{}', start=False):
        self.target = target
        self.replacement = replacement
        self.name = 'replace(%s)' % target
        self.start = start

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        t_toks = self.target.split(' ')
        q_toks = q.rstrip('?.').split(' ')
        replacement_text = self.replacement.format(a)

        for i in range(len(q_toks)):
            if self.start and i != 0:
                continue
            if ' '.join(q_toks[i:i + len(t_toks)]
                        ).rstrip(',').lower() == self.target:
                begin = q_toks[:i]
                end = q_toks[i + len(t_toks):]
                output = ' '.join(begin + [replacement_text] + end)
                if run_fix_style:
                    output = MRCSample.fix_style(output)
                return output
        return None


class FindWHPRule(ConversionRule):
    r"""
    A rule that looks for $WHP's from right to left and does replacements.

    """
    name = 'FindWHP'

    def _recursive_convert(self, node, q, a, tokens, found_whp):
        if node.word:
            return node.word, found_whp
        if not found_whp:
            whp_phrase = MRCSample.convert_whp(node, q, a, tokens)
            if whp_phrase:
                return whp_phrase, True
        child_phrases = []

        for c in node.children[::-1]:
            c_phrase, found_whp = self._recursive_convert(
                c, q, a, tokens, found_whp)
            child_phrases.append(c_phrase)
        out_toks = []

        for i, p in enumerate(child_phrases[::-1]):
            if i == 0 or p.startswith("'"):
                out_toks.append(p)
            else:
                out_toks.append(' ' + p)

        return ''.join(out_toks), found_whp

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        out_phrase, found_whp = self._recursive_convert(
            const_parse, q, a, tokens, False)
        if found_whp:
            if run_fix_style:
                out_phrase = MRCSample.fix_style(out_phrase)
            return out_phrase
        return None


class AnswerRule(ConversionRule):
    r"""
    Just return the answer.

    """
    name = 'AnswerRule'

    def convert(self, q, a, tokens, const_parse, run_fix_style=True):
        return a


CONST_PARSE_MACROS = {
    '$Noun': '$NP/$NN/$NNS/$NNP/$NNPS',
    '$Verb': '$VB/$VBD/$VBP/$VBZ',
    '$Part': '$VBN/$VG',
    '$Be': 'is/are/was/were',
    '$Do': "do/did/does/don't/didn't/doesn't",
    '$WHP': '$WHADJP/$WHADVP/$WHNP/$WHPP',
}

SPECIAL_ALTERATIONS = {
    'States': 'Kingdom',
    'US': 'UK',
    'U.S': 'U.K.',
    'U.S.': 'U.K.',
    'UK': 'US',
    'U.K.': 'U.S.',
    'U.K': 'U.S.',
    'largest': 'smallest',
    'smallest': 'largest',
    'highest': 'lowest',
    'lowest': 'highest',
    'May': 'April',
    'Peyton': 'Trevor',
}

DO_NOT_ALTER = [
    'many',
    'such',
    'few',
    'much',
    'other',
    'same',
    'general',
    'type',
    'record',
    'kind',
    'sort',
    'part',
    'form',
    'terms',
    'use',
    'place',
    'way',
    'old',
    'young',
    'bowl',
    'united',
    'one',
    'ans_mask'
    'likely',
    'different',
    'square',
    'war',
    'republic',
    'doctor',
    'color']
BAD_ALTERATIONS = ['mx2004', 'planet', 'u.s.', 'Http://Www.Co.Mo.Md.Us']

MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july',
          'august', 'september', 'october', 'november', 'december']

CONVERSION_RULES = [
    # Special rules
    ConstituencyRule(
        '$WHP:what $Be $NP called that $VP',
        '{2} that {3} {1} called {1}'),

    # What type of X
    ConstituencyRule(
        "$WHP:what/which type/genre/kind/group of $NP/$Noun $Be $NP",
        '{5} {4} a {1} {3}'),
    ConstituencyRule(
        "$WHP:what/which type/genre/kind/group of $NP/$Noun $Be $VP",
        '{1} {3} {4} {5}'),
    ConstituencyRule(
        "$WHP:what/which type/genre/kind/group of $NP $VP",
        '{1} {3} {4}'),

    # How $JJ
    ConstituencyRule('how $JJ $Be $NP $IN $NP', '{3} {2} {0} {1} {4} {5}'),
    ConstituencyRule('how $JJ $Be $NP $SBAR', '{3} {2} {0} {1} {4}'),
    ConstituencyRule('how $JJ $Be $NP', '{3} {2} {0} {1}'),

    # When/where $Verb
    ConstituencyRule('$WHP:when/where $Do $NP', '{3} occurred in {1}'),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb',
                     '{3} {4} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP/$PP',
                     '{3} {4} {5} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Do $NP $Verb $NP $PP',
                     '{3} {4} {5} {6} in {1}', {4: 'tense-2'}),
    ConstituencyRule('$WHP:when/where $Be $NP', '{3} {2} in {1}'),
    ConstituencyRule(
        '$WHP:when/where $Verb $NP $VP/$ADJP',
        '{3} {2} {4} in {1}'),

    # What/who/how $Do
    ConstituencyRule("$WHP:what/which/who $Do $NP do",
                     '{3} {1}', {0: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb",
                     '{3} {4} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $IN/$NP",
                     '{3} {4} {5} {1}', {4: 'tense-2', 0: 'vbg'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $PP",
                     '{3} {4} {1} {5}', {4: 'tense-2', 0: 'vbg'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb $NP $VP",
                     '{3} {4} {5} {6} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB",
                     '{3} {4} to {5} {1}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who $Do $NP $Verb to $VB $VP",
                     '{3} {4} to {5} {1} {6}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP $Verb $NP $IN $VP",
                     '{3} {4} {5} {6} {1} {7}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP "
                     "$Verb $PP/$S/$VP/$SBAR/$SQ",
                     '{3} {4} {1} {5}', {4: 'tense-2'}),
    ConstituencyRule("$WHP:what/which/who/how $Do $NP "
                     "$Verb $PP $PP/$S/$VP/$SBAR",
                     '{3} {4} {1} {5} {6}', {4: 'tense-2'}),

    # What/who/how $Be
    # Watch out for things that end in a preposition
    ConstituencyRule(
        "$WHP:what/which/who $Be/$MD $NP of $NP $Verb/$Part $IN",
        '{3} of {4} {2} {5} {6} {1}'),
    ConstituencyRule(
        "$WHP:what/which/who $Be/$MD $NP $NP $IN",
        '{3} {2} {4} {5} {1}'),
    ConstituencyRule(
        "$WHP:what/which/who $Be/$MD $NP $VP/$IN",
        '{3} {2} {4} {1}'),
    ConstituencyRule(
        "$WHP:what/which/who $Be/$MD $NP $IN $NP/$VP",
        '{1} {2} {3} {4} {5}'),
    ConstituencyRule(
        '$WHP:what/which/who $Be/$MD $NP $Verb $PP',
        '{3} {2} {4} {1} {5}'),
    ConstituencyRule('$WHP:what/which/who $Be/$MD $NP/$VP/$PP', '{1} {2} {3}'),
    ConstituencyRule("$WHP:how $Be/$MD $NP $VP", '{3} {2} {4} by {1}'),

    # What/who $Verb
    ConstituencyRule("$WHP:what/which/who $VP", '{1} {2}'),

    # $IN what/which $NP
    ConstituencyRule('$IN what/which $NP $Do $NP $Verb $NP',
                     '{5} {6} {7} {1} the {3} of {0}',
                     {1: 'lower', 6: 'tense-4'}),
    ConstituencyRule('$IN what/which $NP $Be $NP $VP/$ADJP',
                     '{5} {4} {6} {1} the {3} of {0}',
                     {1: 'lower'}),
    ConstituencyRule('$IN what/which $NP $Verb $NP/$ADJP $VP',
                     '{5} {4} {6} {1} the {3} of {0}',
                     {1: 'lower'}),
    FindWHPRule(),
]
WHP_RULES = [
    # WHPP rules
    ConstituencyRule(
        '$IN what/which type/sort/kind/group of $NP/$Noun',
        '{1} {0} {4}'),
    ConstituencyRule(
        '$IN what/which type/sort/kind/group of $NP/$Noun $PP',
        '{1} {0} {4} {5}'),
    ConstituencyRule('$IN what/which $NP', '{1} the {3} of {0}'),
    ConstituencyRule('$IN $WP/$WDT', '{1} {0}'),

    # what/which
    ConstituencyRule(
        'what/which type/sort/kind/group of $NP/$Noun',
        '{0} {3}'),
    ConstituencyRule(
        'what/which type/sort/kind/group of $NP/$Noun $PP',
        '{0} {3} {4}'),
    ConstituencyRule('what/which $NP', 'the {2} of {0}'),

    # How many
    ConstituencyRule('how many/much $NP', '{0} {2}'),

    # Replace
    ReplaceRule('what'),
    ReplaceRule('who'),
    ReplaceRule('how many'),
    ReplaceRule('how much'),
    ReplaceRule('which'),
    ReplaceRule('where'),
    ReplaceRule('when'),
    ReplaceRule('why'),
    ReplaceRule('how'),

    # Just give the answer
    AnswerRule(),
]
