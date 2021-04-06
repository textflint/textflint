r"""
Transforms an affirmative sentence into a negative sentence, or vice versa
==========================================================
"""

__all__ = ['ReverseNeg']

from ..transformation import Transformation


class ReverseNeg(Transformation):
    r"""
    Transforms an affirmative sentence into a negative sentence, or vice versa.
    Each sample generate one transformed sample at most.

    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

    def __repr__(self):
        return "ReverseNeg"

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according transform_field.

        :param Sample sample: input data, normally one data component.
        :param str|list field: indicate which field to transform
        :param int n: number of  generated samples
        :return list trans_samples:transformed sample list.
        
        """
        trans_samples = []

        tokens = sample.get_words(field)
        if not tokens:
            return []

        judge_sentence = self._judge_sentence(tokens)

        if judge_sentence == 'remove':
            del_sample = self._get_del_sample(tokens, field, sample)
            if del_sample:
                trans_samples.append(del_sample)

        if judge_sentence == 'add':
            add_sample = self._get_add_sample(field, tokens, sample)
            if add_sample:
                trans_samples.append(add_sample)

        return trans_samples

    @staticmethod
    def _judge_sentence(tokens):
        r"""
        :param tokens: word list
        :return: transformed_text or None
        
        """
        for i in tokens:
            if i in ['not', 'n\'t', 'don', 'didn', 'doesn',
                     'doesn', 'aren', 'isn', 'wasn', 'weren']:
                return 'remove'

        return 'add'

    @staticmethod
    def _check_sentence(tokens):
        """
        Check positive or negative
        
        """
        if len(tokens) < 3:
            return False
        if '?' in tokens:
            return False
        if tokens[0].lower() in ['are', 'is', 'be', 'am', 'was', 'were', 'how',
                                 'why', 'what', 'where', 'who', 'when', 'can',
                                 'do', 'did', 'does', 'could', 'should',
                                 'would', 'will', 'shall', 'thank', 'thanks']:
            return False
        else:
            return True

    def _parse_sentence(self, tokens):
        """
        Dependency Parsing
        """
        sentence = ' '.join(x for x in tokens)
        sentence_tokens = self.processor.sentence_tokenize(sentence)
        root_id_list = []

        parse_tokens = self.processor.get_dep_parser(
            sentence_tokens[0])

        for i, token in enumerate(parse_tokens):
            if len(token) < 4:
                continue
            if token[3] in ['cop', 'ROOT', 'aux']:
                root_id_list.append(i)

        return root_id_list

    def _get_del_sample(self, tokens, field, sample):
        for i, token in enumerate(tokens):
            # do not + verb → verb
            if token in ['do', 'does', 'did'] and len(tokens) > i + 2:
                if tokens[i + 1] in ['not', 'n\'t']:
                    root_id_list = self._parse_sentence(tokens)
                    pos_tag = self.processor.get_pos(tokens[i + 2])[0][1]

                    if pos_tag in [
                            'VB', 'VBP', 'VBZ', 'VBG', 'VBD', 'VBN'] or (
                            i + 2) in root_id_list:
                        del_list = [i, i + 1]
                        del_sample = sample

                        for i, index in enumerate(del_list):
                            del_sample = del_sample.delete_field_at_index(
                                field, index - i)

                        return del_sample

            if token in ['not', 'n\'t', 'don', 'didn', 'doesn', 'doesn',
                         'aren', 'isn', 'wasn', 'weren']:
                return sample.delete_field_at_index(field, i)

        return []

    def _get_add_sample(self, field, tokens, sample):
        root_id_list = self._parse_sentence(tokens)
        if root_id_list:
            check_sentence = self._check_sentence(tokens)
            if check_sentence:
                root_id = root_id_list[0]
                add_sample = self._add_sample(field, tokens, root_id, sample)
                return add_sample
            else:
                return []
        else:
            return []

    def _add_sample(self, field, tokens, root_id, sample):
        if tokens[root_id].lower() in ['is', 'was', 'were', 'am',
                                       'are', '\'s', '\'re', '\'m']:
            add_sample = sample.insert_field_before_index(
                field, root_id + 1, 'not')
            return add_sample

        if tokens[root_id].lower() in ['being']:
            add_sample = sample.insert_field_before_index(
                field, root_id, 'not')
            return add_sample

        if tokens[root_id].lower() in ['do', 'does', 'did', 'can',
                                       'have', 'will', 'could', 'would',
                                       'could', 'should']:
            add_sample = sample.insert_field_before_index(
                field, root_id + 1, 'not')
            return add_sample
        else:
            token_pos = self.processor.get_pos(tokens[root_id])
            trans_sent = []
            if token_pos[0][1] in ['VB', 'VBP', 'VBZ', 'VBG',
                                   'VBD', 'VBN', 'NNS', 'NN']:
                if token_pos[0][1] in ['VB', 'VBP', 'VBG']:
                    neg_word = ['do', 'not']
                if token_pos[0][1] in ['VBD', 'VBN']:
                    neg_word = ['did', 'not']
                else:
                    neg_word = ['does', 'not']
                add_sample = sample

                for i, word in enumerate(neg_word):
                    add_sample = add_sample.insert_field_before_index(
                        field, root_id + i, word)
                return add_sample

            return trans_sent

