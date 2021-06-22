r"""
Swapping words by Mask Language Model
==========================================================
"""

__all__ = ['MLMSuggestion']

import torch
from copy import copy
from collections import defaultdict

from ....common import device as default_device
from ...transformation.word_substitute import WordSubstitute
from ....common.settings import BERT_MODEL_NAME
from ....common.utils.list_op import trade_off_sub_words


class MLMSuggestion(WordSubstitute):
    r"""
    Transforms an input by replacing its tokens with words of mask language
    predicted.
    To accelerate transformation for long text, input single sentence to
    language model rather than whole text.

    """
    def __init__(
        self,
        masked_model=None,
        device=None,
        accrue_threshold=1,
        max_sent_size=100,
        trans_min=1,
        trans_max=10,
        trans_p=0.2,
        stop_words=None,
        **kwargs
    ):
        r"""
        :param str masked_model: masked language model to predicate candidates
        :param str device: indicate utilize cpu or which gpu device to run
            neural network
        :param int accrue_threshold: threshold of Bert results to pick
        :param max_sent_size: max_sent_size
        :param int trans_min: Minimum number of character will be augmented.
        :param int trans_max: Maximum number of character will be augmented.
            If None is passed, number of augmentation is calculated via aup_char_p.
            If calculated result from aug_p is smaller than aug_max, will use
            calculated result from aup_char_p. Otherwise, using aug_max.
        :param float trans_p: Percentage of character (per token) will be
            augmented.
        :param list stop_words: List of words which will be skipped from augment
            operation.

        """
        super().__init__(
            trans_min=trans_min,
            trans_max=trans_max,
            trans_p=trans_p,
            stop_words=stop_words
        )
        self.device = self.get_device(device) if device else default_device
        self.max_sent_size = max_sent_size
        self.get_pos = True
        self.accrue_threshold = accrue_threshold
        self.masked_model = masked_model if masked_model else BERT_MODEL_NAME
        self.tokenizer = None
        self.model = None
        self.pos_allowed_token_id = None

    def __repr__(self):
        return 'MLMSuggestion'

    @staticmethod
    def get_device(device):
        r"""
        Get gpu or cpu device.
        :param str device: device string
                           "cpu" means use cpu device.
                           "cuda:0" means use gpu device which index is 0.
        :return: device in torch.
        """
        if "cuda" not in device:
            return torch.device("cpu")
        else:
            return torch.device(device)

    def get_model(self):
        r"""
        Loads masked language model to predict candidates.

        """
        from transformers import BertTokenizer, BertForMaskedLM

        self.tokenizer = BertTokenizer.from_pretrained(
            self.masked_model, do_lower_case=False)
        self.model = BertForMaskedLM.from_pretrained(self.masked_model)
        self.model.to(self.device)
        self.model.eval()

    def pre_calculate_allowed_tokens(self):
        r"""
        Precalculate meaningful tokens, filter tokens which is not an alphabetic
        string.

        Pre filter would accelerate procedure of verifying pos tags of
        candidates.

        """
        pos_to_token_id_dict = defaultdict(list)

        bert_tokens = self.tokenizer.convert_ids_to_tokens(
            list(range(self.tokenizer.vocab_size))
        )
        bert_token_pos = self.processor.get_pos(bert_tokens)

        for index, pos in enumerate(bert_token_pos):
            word = bert_tokens[index]

            if not word.isalpha() or len(word) == 1:
                continue

            pos_to_token_id_dict[pos[1][:2]].append(index)

        for pos, indices in pos_to_token_id_dict.items():
            pos_to_token_id_dict[pos] = torch.tensor(
                indices, dtype=torch.long, device=self.device
            )

        return pos_to_token_id_dict

    def _transform(self, sample, field='x', n=1, **kwargs):
        r"""
        Transform text string according field.

        :param Sample sample: input data, normally one data component.
        :param str field: indicate which field to apply transformation
        :param int n: number of generated samples
        :param kwargs:
        :return list trans_samples: transformed sample list.

        """

        if not self.pos_allowed_token_id:
            self.get_model()
            self.pos_allowed_token_id = self.pre_calculate_allowed_tokens()

        tokens = sample.get_words(field)
        tokens_mask = sample.get_mask(field)
        # accelerate computation for long text
        if len(tokens) > self.max_sent_size:
            sentences = sample.get_sentences(field)
            sentences_tokens = [
                self.processor.tokenize(sent) for sent in sentences
            ]
        else:
            sentences_tokens = [tokens]

        # return up to (len(sub_indices) * n) candidates
        pos_info = sample.get_pos(field)
        legal_indices = self.skip_aug(tokens, tokens_mask)

        if not legal_indices:
            return []

        sub_words, sub_indices = self._get_substitute_words(
            tokens, legal_indices, sentences_tokens, pos=pos_info, n=n
        )

        # select property candidates
        sub_words, sub_indices = trade_off_sub_words(
            sub_words, sub_indices, n=n
        )

        if not sub_words:
            return []

        trans_samples = []

        for i in range(len(sub_words)):
            single_sub_words = sub_words[i]
            trans_samples.append(
                sample.replace_field_at_indices(
                    field, sub_indices, single_sub_words))

        return trans_samples

    def _get_substitute_words(self, words, legal_indices,
                              sentences_tokens, pos=None, n=5):
        r"""
        Returns a list containing all possible words .

        Overwrite _get_substitute_words of super class.
        To accelerate transformation for long text, input single sentence to
        language model rather than whole text.

        :param list words: all words
        :param list legal_indices: indices which has not been skipped
        :param list sentences_tokens: list of tokens of each sentence
        :param list|None pos: None or list of pos tags
        :param int n:max candidates for each word to be substituted
        :return list candidates_list: list of candidates list
        :return list candidates_indices: list of candidates_indices list

        """
        sub_indices, sub_sentences, sub_sent_indices = \
            self._get_relate_sub_info(words, sentences_tokens, legal_indices)

        assert len(sub_sentences) == len(sub_indices)

        candidates_list = []
        candidates_indices = []
        mask_word_pos_list = []
        mask_indices = []
        batch_tokens_tensor = torch.tensor(
            [], dtype=torch.long, device=self.device
        )
        batch_size = 0

        for i, mask_word_index in enumerate(sub_indices):
            mask_original_word_pos = pos[mask_word_index][:2]
            if mask_original_word_pos not in self.pos_allowed_token_id:
                continue

            candidates_indices.append(mask_word_index)
            mask_indices.append(sub_sent_indices[i] + 1)
            mask_word_pos_list.append(mask_original_word_pos)

            # process each legal words to get maximum transformed samples
            sent_tokens = ['[CLS]'] + \
                copy(sentences_tokens[sub_sentences[i]]) + ['[SEP]']
            sent_tokens[sub_sent_indices[i] + 1] = '[MASK]'
            # padding sentence to do batch predict
            sent_tokens = sent_tokens + ['[PAD]'] * \
                (self.max_sent_size - len(sent_tokens))
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(
                sent_tokens[:self.max_sent_size]
            )
            tokens_tensor = torch.tensor(
                [indexed_tokens], dtype=torch.long, device=self.device
            )
            batch_tokens_tensor = torch.cat(
                (batch_tokens_tensor, tokens_tensor)
            )
            batch_size += 1

        if batch_size > 0:
            segments_tensors = torch.zeros(
                batch_size,
                self.max_sent_size,
                dtype=torch.int64,
                device=self.device
            )
            candidates_list = self._get_candidates(
                batch_tokens_tensor,
                segments_tensors,
                mask_indices,
                mask_word_pos_list,
                n=n
            )

        return candidates_list, candidates_indices

    def _get_relate_sub_info(self, words, sentences_tokens, legal_indices):
        r"""
        Get indices of substitute words.

        :param list words: original tokens without sentence split
        :param list sentences_tokens: sentence token lists split from words
        :param list legal_indices: legal indices which are allowed substituted
        :return: list sub_indices, sub_sentences, sub_sent_indices
        """
        sentences_indices = []
        idx = 0

        for sentence_tokens in sentences_tokens:
            sentences_indices.append(
                list(range(idx, idx + len(sentence_tokens)))
            )
            idx += len(sentence_tokens)

        trans_num = self.get_trans_cnt(len(words))
        sub_indices = sorted(self.sample_num(legal_indices, trans_num))
        valid_sub_indices = []
        sub_sentences = []
        sub_sent_indices = []

        for sub_index in sub_indices:
            for idx, sentence_indices in enumerate(sentences_indices):
                # skip out of index cases
                if sub_index in sentence_indices and \
                        sentence_indices.index(sub_index) < self.max_sent_size:
                    valid_sub_indices.append(sub_index)
                    sub_sentences.append(idx)
                    sub_sent_indices.append(sentence_indices.index(sub_index))

        return valid_sub_indices, sub_sentences, sub_sent_indices

    def _get_candidates(self, batch_tokens_tensor, segments_tensors,
                        mask_indices, mask_word_pos_list, n=5):
        r"""
        Get candidates from MLM model.

        :param torch.tensor batch_tokens_tensor: tokens tensor input
        :param torch.tensor segments_tensors: segment input
        :param list mask_indices: indices to predict candidates
        :param list mask_word_pos_list: pos tags of original target words
        :param int n: candidates number
        :return: list candidates
        """
        with torch.no_grad():
            output = self.model(batch_tokens_tensor, segments_tensors)

        candidates_list = []

        for i, tup in enumerate(zip(mask_indices, mask_word_pos_list)):
            mask_index, mask_original_word_pos = tup
            predict_tensor = output[0][i, mask_index]
            allowed_token_id = self.pos_allowed_token_id[mask_original_word_pos]
            pos_allowed_predict = predict_tensor.gather(0, allowed_token_id)
            prob_values, topk_index = pos_allowed_predict.topk(
                min(pos_allowed_predict.shape[0], n)
            )
            original_vocab_index = allowed_token_id.gather(0, topk_index)
            replace_words = self.tokenizer.convert_ids_to_tokens(
                original_vocab_index
            )
            candidates_list.append(replace_words)

        return candidates_list

    def skip_aug(self, tokens, mask, pos=None):
        return self.pre_skip_aug(tokens, mask)
