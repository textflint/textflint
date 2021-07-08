r"""
Use Bert to generate words.
==========================================================
"""
__all__ = ["CnMLM"]
import torch
from transformers import BertTokenizer, BertForMaskedLM
from ..transformation import Transformation
from ....input.component.field.cn_text_field import CnTextField
from ....common.preprocess.cn_processor import CnProcessor
from ....common.settings import ORIGIN, MODIFIED_MASK


class CnMLM(Transformation):
    r"""
    Use Bert to generate words.

    Example::

        小明喜欢看书 -> 小明喜欢看报纸

    """

    def __init__(self, **kwargs):
        r"""
        :param **kwargs:
        """
        super().__init__()

    def __repr__(self):
        return 'CnMLM'

    def _transform(self, sample, n=1, **kwargs):
        r"""
        In this function, because there is only one deformation mode, only one
        set of outputs is output.

        :param ~textflint.CWSSample sample: the data which need be changed
        :param **kwargs:
        :return: trans_sample a list of sample

        """
        # get sentence label and pos tag
        origin_sentence = sample.get_value('x')
        origin_label = sample.get_value('y')
        pos_tags = sample.pos_tags
        x, y, mask = self._get_transformations(
            origin_sentence, origin_label, pos_tags, sample.mask)
        if x == origin_sentence:
            return []
        x = CnTextField(x, mask)
        return [sample.update(x, y)]

    def create_word(self, sentence):
        r"""
        Crete the word we need

        :param str sentence: the sentence with [MASK]
        :return: the change sentence

        """
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        text = '[CLS] ' + sentence
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        # Create the segments tensors.
        segments_ids = [0] * len(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Load pre-trained model (weights)
        model = BertForMaskedLM.from_pretrained('bert-base-chinese')
        model.eval()
        masked_index = tokenized_text.index('[MASK]')
        masked_index1 = masked_index + 1
        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)

        predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
        predicted_index1 = torch.argmax(
            predictions[0][0][masked_index1]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        predicted_token1 = tokenizer.convert_ids_to_tokens([predicted_index1])[
            0]
        # Determine whether the generated words meet the requirements
        if len(predicted_token) != 1 or len(predicted_token1) != 1 or \
                self.is_word(predicted_token + predicted_token1):
            return ''
        # Change the generated sentence
        return predicted_token + predicted_token1

    def _get_transformations(self, sentence, label, pos_tags, mask):
        r"""
        Generate word function.

        :param str sentence: chinese sentence
        :param list label: Chinese word segmentation tag
        :param list pos_tags: sentence's pos tag
        :return list: two list include the pos and labels which are changed

        """
        assert len(sentence) == len(label)
        cnt = 0
        for i in range(len(pos_tags)):
            tag, start, end = pos_tags[i]
            start += cnt
            end += cnt
            # find the pos that can generate word
            # Situation 1: v + single n
            # we generate double n replace single n
            if label[start] == 'B' and label[start + 1] == 'E' and \
                    i < len(pos_tags) - 1 and pos_tags[i][0] == 'v' \
                    and end == start + 1 and \
                    self.check_part_pos(sentence[start + 1]):
                token = ''
                for j in range(len(sentence)):
                    if j != start + 1:
                        token += sentence[j] + ' '
                    else:
                        token += '[MASK] [MASK] '
                change = self.create_word(token)
                if change != '':
                    if self.check(start, end, mask):
                        sentence = sentence[:start + 1] + \
                            change + sentence[start + 2:]
                        label = label[:start] + \
                            ['S', 'B', 'E'] + label[start + 2:]
                        mask = mask[:start + 1] + \
                            [MODIFIED_MASK] * 2 + mask[start + 2:]
                        cnt += 1
                        start += 1
            # Situation 1: n + n + n
            # we generate double n replace single n and split one word into two
            elif label[start:start + 3] == ['B', 'M', 'E'] and \
                    tag == 'n' and end - start == 2:
                token = ''
                start += 2
                for i in range(len(sentence)):
                    if i != start:
                        token += sentence[i] + ' '
                    else:
                        token += '[MASK] [MASK] '
                change = self.create_word(token)
                if self.check(start, end, mask):
                    if change != '':
                        sentence = sentence[:start] + \
                            change + sentence[start + 1:]
                        label = label[:start - 1] + \
                            ['E', 'B', 'E'] + label[start + 1:]
                        mask = mask[:start] + [MODIFIED_MASK] * \
                            2 + mask[start + 1:]
                        cnt += 1
                        start += 1
            start += 1

        return sentence, label, mask

    @staticmethod
    def is_word(sentence):
        from ltp import LTP
        r""" 
        Judge whether it is a word.

        :param str sentence: input sentence string
            sentence: input sentence string
        :return bool: is a word or not
        
        """
        if sentence[0] == sentence[1]:
            return True
        ltp = LTP()
        seg, hidden = ltp.seg([sentence])
        pos = ltp.pos(hidden)
        pos = pos[0]
        if len(pos) == 1 and pos[0] == 'n':
            return False
        return True

    @staticmethod
    def check(start, end, mask):
        for i in range(start, end + 1):
            if mask[i] != ORIGIN:
                return False
        return True

    @staticmethod
    def check_part_pos(sentence):
        """
        get the pos of sentence if we need

        :param str sentence: origin word
        :return: bool
        """
        if sentence == "":
            return False
        Processor = CnProcessor()
        pos = Processor.get_pos_tag(sentence)
        if len(pos) == 1 and pos[0][0] == 'n':
            return True
        return False
