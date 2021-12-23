r"""
MRC Sample Class
==========================================================
Manage text transformation for MRC.
Heavily borrowed from adversarial-squad.
For code in adversarial-squad, please check the following link:
https://github.com/robinjia/adversarial-squad
"""

from .cnsample import CnSample
from ..field import Field, CnTextField
from ....common.utils.list_op import normalize_scope
from ....common.preprocess.nltk_res_load import ModelManager
from ....common.settings import *

from nltk.stem.lancaster import LancasterStemmer
from copy import deepcopy


__all__ = ['MRCCnSample']


class MRCCnSample(CnSample):
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
        return 'MRCCnSample'

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

        for answer in self.answers:
            answer_text =  self.context.text[answer['start']:answer['end'] + 1]
            if answer['text'] != answer_text:
                return False

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
        Convert data dict which contains essential information to MRCCnSample.

        :param dict data: the dict obj that contains dict info

        """
        self.context = CnTextField(data['context'])
        self.question = CnTextField(data['question'])
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
            spans = self.convert_idx(data['context'], self.context.tokens)
            self.load_answers(data['answers'], spans)

            if not self.is_legal():
                raise ValueError("Data sample {0} is not legal, "
                                 "Answer spans mismatch answer text."
                                 .format(data))

    def dump(self):
        r"""
        Convert data dict which contains essential information to MRCCnSample.

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
                elif scope[1] >= answer['start'] and scope[1] <= answer['end']:
                    answer['end'] -= offset
                    answer['text'] = self.context.text[answer['start']:answer['end']+1]
        sample = super(MRCCnSample, self).delete_field_at_indices(field, indices)
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
                items[i] = MRCCnSample.text_processor.tokenize(items[i],cws=False)
                offset = len(items[i])

            for answer in answers:
                if index <= answer['start']:
                    answer['start'] += offset
                    answer['end'] += offset
        sample = super(MRCCnSample, self).insert_field_before_indices(
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
                items[i] = MRCCnSample.text_processor.tokenize(items[i],cws =False)
                offset = len(items[i])

            for answer in answers:
                if index < answer['start']:
                    answer['start'] += offset
                    answer['end'] += offset
        sample = super(MRCCnSample, self).insert_field_after_indices(
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



    def convert_answer(self, answer, sent_tokens, question):
        """
        Replace the ground truth with fake answer based on specific rules

        :param str answer: ground truth, str
        :param list sent_tokens: sentence dicts, like [{'word': 'Saint',
            'pos': 'NNP', 'lemma': 'Saint', 'ner': 'PERSON'}...]
        :param str question: question sentence
        :return str: fake answer

        """
        tokens = MRCCnSample.get_answer_tokens(sent_tokens, answer)
        determiner = MRCCnSample.get_determiner_for_answers(answer)
        for rule_name, func in self.get_answer_rules():
            new_ans = func(answer, tokens, question, determiner=determiner)
            if new_ans:
                return new_ans
        return None
