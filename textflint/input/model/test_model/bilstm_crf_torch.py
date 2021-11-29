"""
biLSTM-crf for NER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
import argparse
import codecs
import math
import os
import re
import string

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def log_sum_exp(x):
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))

def get_words_num(word_sequences):
    return sum(len(word_seq) for word_seq in word_sequences)

class DatasetsBank():
    """DatasetsBank provides storing the train/dev/test data subsets and sampling batches from the train dataset."""
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.unique_words_list = list()

    def __add_to_unique_words_list(self, word_sequences):
        for word_seq in word_sequences:
            for word in word_seq:
                if word not in self.unique_words_list:
                    self.unique_words_list.append(word)
        if self.verbose:
            print('DatasetsBank: len(unique_words_list) = %d unique words.' % (len(self.unique_words_list)))

    def add_train_sequences(self, word_sequences_train, tag_sequences_train):
        self.train_data_num = len(word_sequences_train)
        self.word_sequences_train = word_sequences_train
        self.tag_sequences_train = tag_sequences_train
        self.__add_to_unique_words_list(word_sequences_train)

    def add_dev_sequences(self, word_sequences_dev, tag_sequences_dev):
        self.word_sequences_dev = word_sequences_dev
        self.tag_sequences_dev = tag_sequences_dev
        self.__add_to_unique_words_list(word_sequences_dev)

    def add_test_sequences(self, word_sequences_test, tag_sequences_test):
        self.word_sequences_test = word_sequences_test
        self.tag_sequences_test = tag_sequences_test
        self.__add_to_unique_words_list(word_sequences_test)

    def __get_train_batch(self, batch_indices):
        word_sequences_train_batch = [self.word_sequences_train[i] for i in batch_indices]
        tag_sequences_train_batch = [self.tag_sequences_train[i] for i in batch_indices]
        return word_sequences_train_batch, tag_sequences_train_batch

    def get_train_batches(self, batch_size):
        random_indices = np.random.permutation(np.arange(self.train_data_num))
        for k in range(self.train_data_num // batch_size): # oh yes, we drop the last batch
            batch_indices = random_indices[k:k + batch_size].tolist()
            word_sequences_train_batch, tag_sequences_train_batch = self.__get_train_batch(batch_indices)
            yield word_sequences_train_batch, tag_sequences_train_batch

class SeqIndexerBase():
    """
    SeqIndexerBase is a base abstract class for sequence indexers. It converts list of lists of string items
    to the list of lists of integer indices and back. Items could be either words, tags or characters.
    """
    def __init__(self, gpu=-1, check_for_lowercase=True, zero_digits=False, pad='<pad>', unk='<unk>',
                 load_embeddings=False, embeddings_dim=0, verbose=False):
        self.gpu = gpu
        self.check_for_lowercase = check_for_lowercase
        self.zero_digits = zero_digits
        self.pad = pad
        self.unk = unk
        self.load_embeddings = load_embeddings
        self.embeddings_dim = embeddings_dim
        self.verbose = verbose
        self.out_of_vocabulary_list = list()
        self.item2idx_dict = dict()
        self.idx2item_dict = dict()
        if load_embeddings:
            self.embeddings_loaded = False
            self.embedding_vectors_list = list()
        if pad is not None:
            self.pad_idx = self.add_item(pad)
            if load_embeddings:
                self.add_emb_vector(self.generate_zero_emb_vector())
        if unk is not None:
            self.unk_idx = self.add_item(unk)
            if load_embeddings:
                self.add_emb_vector(self.generate_random_emb_vector())

    def get_items_list(self):
        return list(self.item2idx_dict.keys())

    def get_items_count(self):
        return len(self.get_items_list())

    def item_exists(self, item):
        return item in self.item2idx_dict.keys()

    def add_item(self, item):
        idx = len(self.get_items_list())
        self.item2idx_dict[item] = idx
        self.idx2item_dict[idx] = item
        return idx

    def get_class_num(self):
        if self.pad is not None and self.unk is not None:
            return self.get_items_count() - 2
        if self.pad is not None or self.unk is not None:
            return self.get_items_count() - 1
        return self.get_items_count()

    def items2idx(self, item_sequences):
        idx_sequences = []
        for item_seq in item_sequences:
            idx_seq = list()
            for item in item_seq:
                if item in self.item2idx_dict:
                    idx_seq.append(self.item2idx_dict[item])
                else:
                    if self.unk is not None:
                        idx_seq.append(self.item2idx_dict[self.unk])
                    else:
                        idx_seq.append(self.item2idx_dict[self.pad])
            idx_sequences.append(idx_seq)
        return idx_sequences

    def idx2items(self, idx_sequences):
        item_sequences = []
        for idx_seq in idx_sequences:
            item_seq = [self.idx2item_dict[idx] for idx in idx_seq]
            item_sequences.append(item_seq)
        return item_sequences

    def items2tensor(self, item_sequences, align='left', word_len=-1):
        idx = self.items2idx(item_sequences)
        return self.idx2tensor(idx, align, word_len)

    def idx2tensor(self, idx_sequences, align='left', word_len=-1):
        batch_size = len(idx_sequences)
        if word_len == -1:
            word_len = max([len(idx_seq) for idx_seq in idx_sequences])
        tensor = torch.zeros(batch_size, word_len, dtype=torch.long)
        #if self.gpu >= 0:
        #    tensor = torch.cuda.LongTensor(batch_size, word_len).fill_(0)
        #else:
        #    tensor = torch.LongTensor(batch_size, word_len).fill_(0)
        for k, idx_seq in enumerate(idx_sequences):
            curr_seq_len = len(idx_seq)
            if curr_seq_len > word_len:
                idx_seq = [idx_seq[i] for i in range(word_len)]
                curr_seq_len = word_len
            if align == 'left':
                tensor[k, :curr_seq_len] = torch.LongTensor(np.asarray(idx_seq))
            elif align == 'center':
                start_idx = (word_len - curr_seq_len) // 2
                tensor[k, start_idx:start_idx+curr_seq_len] = torch.LongTensor(np.asarray(idx_seq))
            else:
                raise ValueError('Unknown align string.')
        if self.gpu >= 0:
            tensor = tensor.cuda(device=self.gpu)
        return tensor

class SeqIndexerTag(SeqIndexerBase):
    """SeqIndexerTag converts list of lists of string tags to list of lists of integer indices and back."""
    def __init__(self, gpu):
        SeqIndexerBase.__init__(self, gpu=gpu, check_for_lowercase=False, zero_digits=False,
                                      pad='<pad>', unk=None, load_embeddings=False, verbose=True)

    def add_tag(self, tag):
        if not self.item_exists(tag):
            self.add_item(tag)

    def load_items_from_tag_sequences(self, tag_sequences):
        assert self.load_embeddings == False
        for tag_seq in tag_sequences:
            for tag in tag_seq:
                self.add_tag(tag)
        if self.verbose:
            print('\nload_vocabulary_from_tag_sequences:')
            print(' -- class_num = %d' % self.get_class_num())
            print(' --', self.item2idx_dict)

class SeqIndexerBaseEmbeddings(SeqIndexerBase):
    """
    SeqIndexerBaseEmbeddings is a basic abstract sequence indexers class that implements work qith embeddings.
    """
    def __init__(self, gpu, check_for_lowercase, zero_digits, pad, unk, load_embeddings, embeddings_dim, verbose):
        SeqIndexerBase.__init__(self, gpu, check_for_lowercase, zero_digits, pad, unk, load_embeddings, embeddings_dim,
                                verbose)
    @staticmethod
    def load_embeddings_from_file(emb_fn, emb_delimiter, verbose=True):
        for k, line in enumerate(open(emb_fn, 'r', encoding='utf-8')):
            values = line.split(emb_delimiter)
            if len(values) < 5:
                continue
            word = values[0]
            emb_vector = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), values[1:])))
            if verbose:
                if k % 25000 == 0:
                    print('Reading embeddings file %s, line = %d' % (emb_fn, k))
            yield word, emb_vector

    def generate_zero_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return [0 for _ in range(self.embeddings_dim)]

    def generate_random_emb_vector(self):
        if self.embeddings_dim == 0:
            raise ValueError('embeddings_dim is not known.')
        return np.random.uniform(-np.sqrt(3.0 / self.embeddings_dim), np.sqrt(3.0 / self.embeddings_dim),
                                 self.embeddings_dim).tolist()

    def add_emb_vector(self, emb_vector):
        self.embedding_vectors_list.append(emb_vector)

    def get_loaded_embeddings_tensor(self):
        return torch.FloatTensor(np.asarray(self.embedding_vectors_list))

class SeqIndexerWord(SeqIndexerBaseEmbeddings):
    """SeqIndexerWord converts list of lists of words as strings to list of lists of integer indices and back."""
    def __init__(self, gpu=-1, check_for_lowercase=True, embeddings_dim=0, verbose=True):
        SeqIndexerBaseEmbeddings.__init__(self, gpu=gpu, check_for_lowercase=check_for_lowercase, zero_digits=True,
                                          pad='<pad>', unk='<unk>', load_embeddings=True, embeddings_dim=embeddings_dim,
                                          verbose=verbose)
        self.original_words_num = 0
        self.lowercase_words_num = 0
        self.zero_digits_replaced_num = 0
        self.zero_digits_replaced_lowercase_num = 0
        self.capitalize_word_num = 0
        self.uppercase_word_num = 0

    def load_items_from_embeddings_file_and_unique_words_list(self, emb_fn, emb_delimiter, emb_load_all,
                                                              unique_words_list):
        # Get the full list of available case-sensitive words from text file with pretrained embeddings
        embeddings_words_list = [emb_word for emb_word, _ in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn,
                                                                                                          emb_delimiter,
                                                                                                          verbose=True)]
        # Create reverse mapping word from the embeddings file -> list of unique words from the dataset
        emb_word_dict2unique_word_list = dict()
        out_of_vocabulary_words_list = list()
        for unique_word in unique_words_list:
            emb_word = self.get_embeddings_word(unique_word, embeddings_words_list)
            if emb_word is None:
                out_of_vocabulary_words_list.append(unique_word)
            else:
                if emb_word not in emb_word_dict2unique_word_list:
                    emb_word_dict2unique_word_list[emb_word] = [unique_word]
                else:
                    emb_word_dict2unique_word_list[emb_word].append(unique_word)
        # Add pretrained embeddings for unique_words
        for emb_word, emb_vec in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn, emb_delimiter,verbose=True):
            if emb_word in emb_word_dict2unique_word_list:
                for unique_word in emb_word_dict2unique_word_list[emb_word]:
                    self.add_word_emb_vec(unique_word, emb_vec)
        if self.verbose:
            print('\nload_vocabulary_from_embeddings_file_and_unique_words_list:')
            print('    First 50 OOV words:')
            for i, oov_word in enumerate(out_of_vocabulary_words_list):
                print('        out_of_vocabulary_words_list[%d] = %s' % (i, oov_word))
                if i > 49:
                    break
            print(' -- len(out_of_vocabulary_words_list) = %d' % len(out_of_vocabulary_words_list))
            print(' -- original_words_num = %d' % self.original_words_num)
            print(' -- lowercase_words_num = %d' % self.lowercase_words_num)
            print(' -- zero_digits_replaced_num = %d' % self.zero_digits_replaced_num)
            print(' -- zero_digits_replaced_lowercase_num = %d' % self.zero_digits_replaced_lowercase_num)
        # Load all embeddings
        if emb_load_all:
            loaded_words_list = self.get_items_list()
            load_all_words_num_before = len(loaded_words_list)
            load_all_words_lower_num = 0
            load_all_words_upper_num = 0
            load_all_words_capitalize_num = 0
            for emb_word, emb_vec in SeqIndexerBaseEmbeddings.load_embeddings_from_file(emb_fn, emb_delimiter,                                                                                        verbose=True):
                if emb_word in loaded_words_list:
                    continue
                if emb_word.lower() not in loaded_words_list and emb_word.lower() not in embeddings_words_list:
                    self.add_word_emb_vec(emb_word.lower(), emb_vec)
                    load_all_words_lower_num += 1
                if emb_word.upper() not in loaded_words_list and emb_word.upper() not in embeddings_words_list:
                    self.add_word_emb_vec(emb_word.upper(), emb_vec)
                    load_all_words_upper_num += 1
                if emb_word.capitalize() not in loaded_words_list and emb_word.capitalize() not in \
                        embeddings_words_list:
                    self.add_word_emb_vec(emb_word.capitalize(), emb_vec)
                    load_all_words_capitalize_num += 1
                self.add_item(emb_word)
                self.add_emb_vector(emb_vec)
            load_all_words_num_after = len(self.get_items_list())
            if self.verbose:
                print(' ++ load_all_words_num_before = %d ' % load_all_words_num_before)
                print(' ++ load_all_words_lower_num = %d ' % load_all_words_lower_num)
                print(' ++ load_all_words_num_after = %d ' % load_all_words_num_after)

    def get_embeddings_word(self, word, embeddings_word_list):
        if word in embeddings_word_list:
            self.original_words_num += 1
            return word
        elif self.check_for_lowercase and word.lower() in embeddings_word_list:
            self.lowercase_words_num += 1
            return word.lower()
        elif self.zero_digits and re.sub('\d', '0', word) in embeddings_word_list:
            self.zero_digits_replaced_num += 1
            return re.sub('\d', '0', word)
        elif self.check_for_lowercase and self.zero_digits and re.sub('\d', '0', word.lower()) in embeddings_word_list:
            self.zero_digits_replaced_lowercase_num += 1
            return re.sub('\d', '0', word.lower())
        return None

    def add_word_emb_vec(self, word, emb_vec):
        self.add_item(word)
        self.add_emb_vector(emb_vec)

    def get_unique_characters_list(self, verbose=False, init_by_printable_characters=True):
        if init_by_printable_characters:
            unique_characters_set = set(string.printable)
        else:
            unique_characters_set = set()
        if verbose:
            cnt = 0
        for n, word in enumerate(self.get_items_list()):
            len_delta = len(unique_characters_set)
            unique_characters_set = unique_characters_set.union(set(word))
            if verbose and len(unique_characters_set) > len_delta:
                cnt += 1
                print('n = %d/%d (%d) %s' % (n, len(self.get_items_list), cnt, word))
        return list(unique_characters_set)

class LayerBase(nn.Module):
    """Abstract base class for all type of layers."""
    def __init__(self, gpu):
        super(LayerBase, self).__init__()
        self.gpu = gpu

    def tensor_ensure_gpu(self, tensor):
        if self.is_cuda():
            return tensor.cuda(device=self.gpu)
        else:
            return tensor.cpu()

    def apply_mask(self, input_tensor, mask_tensor):
        input_tensor = self.tensor_ensure_gpu(input_tensor)
        mask_tensor = self.tensor_ensure_gpu(mask_tensor)
        return input_tensor*mask_tensor.unsqueeze(-1).expand_as(input_tensor)

    def get_seq_len_list_from_mask_tensor(self, mask_tensor):
        batch_size = mask_tensor.shape[0]
        return [int(mask_tensor[k].sum().item()) for k in range(batch_size)]

class LayerBiRNNBase(LayerBase):
    """LayerBiRNNBase is abstract base class for all bidirectional recurrent layers."""
    def __init__(self, input_dim, hidden_dim, gpu):
        super(LayerBiRNNBase, self).__init__(gpu)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2

    def sort_by_seq_len_list(self, seq_len_list):
        data_num = len(seq_len_list)
        sort_indices = sorted(range(len(seq_len_list)), key=seq_len_list.__getitem__, reverse=True)
        reverse_sort_indices = [-1 for _ in range(data_num)]
        for i in range(data_num):
            reverse_sort_indices[sort_indices[i]] = i
        sort_index = self.tensor_ensure_gpu(torch.tensor(sort_indices, dtype=torch.long))
        reverse_sort_index = self.tensor_ensure_gpu(torch.tensor(reverse_sort_indices, dtype=torch.long))
        return sorted(seq_len_list, reverse=True), sort_index, reverse_sort_index

    def pack(self, input_tensor, mask_tensor):
        seq_len_list = self.get_seq_len_list_from_mask_tensor(mask_tensor)
        sorted_seq_len_list, sort_index, reverse_sort_index = self.sort_by_seq_len_list(seq_len_list)
        input_tensor_sorted = torch.index_select(input_tensor, dim=0, index=sort_index)
        return pack_padded_sequence(input_tensor_sorted, lengths=sorted_seq_len_list, batch_first=True), \
               reverse_sort_index

    def unpack(self, output_packed, max_seq_len, reverse_sort_index):
        output_tensor_sorted, _ = pad_packed_sequence(output_packed, batch_first=True, total_length=max_seq_len)
        output_tensor = torch.index_select(output_tensor_sorted, dim=0, index=reverse_sort_index)
        return output_tensor

class LayerWordEmbeddings(LayerBase):
    """LayerWordEmbeddings implements word embeddings."""
    def __init__(self, word_seq_indexer, gpu, freeze_word_embeddings=False, pad_idx=0):
        super(LayerWordEmbeddings, self).__init__(gpu)
        embeddings_tensor = word_seq_indexer.get_loaded_embeddings_tensor()
        self.embeddings = nn.Embedding.from_pretrained(embeddings=embeddings_tensor, freeze=freeze_word_embeddings)
        self.embeddings.padding_idx = pad_idx
        self.word_seq_indexer = word_seq_indexer
        self.freeze_embeddings = freeze_word_embeddings
        self.embeddings_num = embeddings_tensor.shape[0]
        self.embeddings_dim = embeddings_tensor.shape[1]
        self.output_dim = self.embeddings_dim

    def is_cuda(self):
        return self.embeddings.weight.is_cuda

    def forward(self, word_sequences):
        input_tensor = self.tensor_ensure_gpu(self.word_seq_indexer.items2tensor(word_sequences)) # shape: batch_size x max_seq_len
        word_embeddings_feature = self.embeddings(input_tensor) # shape: batch_size x max_seq_len x output_dim
        return word_embeddings_feature

class LayerBiLSTM(LayerBiRNNBase):
    """BiLSTM layer implements standard bidirectional LSTM recurrent layer"""
    def __init__(self, input_dim, hidden_dim, gpu):
        super(LayerBiLSTM, self).__init__(input_dim, hidden_dim, gpu)
        self.num_layers = 1
        self.num_directions = 2
        rnn = nn.LSTM(input_size=input_dim,
                      hidden_size=hidden_dim,
                      num_layers=1,
                      batch_first=True,
                      bidirectional=True)
        self.rnn = rnn

    def lstm_custom_init(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse)
        self.rnn.bias_hh_l0.data.fill_(0)
        self.rnn.bias_hh_l0_reverse.data.fill_(0)
        self.rnn.bias_ih_l0.data.fill_(0)
        self.rnn.bias_ih_l0_reverse.data.fill_(0)
        # Init forget gates to 1
        for names in self.rnn._all_weights:
            for name in filter(lambda n: 'bias' in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def forward(self, input_tensor, mask_tensor): #input_tensor shape: batch_size x max_seq_len x dim
        batch_size, max_seq_len, _ = input_tensor.shape
        input_packed, reverse_sort_index = self.pack(input_tensor, mask_tensor)
        h0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        c0 = self.tensor_ensure_gpu(torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim))
        output_packed, _ = self.rnn(input_packed, (h0, c0))
        output_tensor = self.unpack(output_packed, max_seq_len, reverse_sort_index)
        return output_tensor  # shape: batch_size x max_seq_len x hidden_dim*2

    def is_cuda(self):
        return self.rnn.weight_hh_l0.is_cuda

class LayerCRF(LayerBase):
    """LayerCRF implements Conditional Random Fields (Ma.et.al., 2016 style)"""
    def __init__(self, gpu, states_num, pad_idx, sos_idx, tag_seq_indexer, verbose=True):
        super(LayerCRF, self).__init__(gpu)
        self.states_num = states_num
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.tag_seq_indexer = tag_seq_indexer
        self.tag_seq_indexer.add_tag('<sos>')
        self.verbose = verbose
        # Transition matrix contains log probabilities from state j to state i
        self.transition_matrix = nn.Parameter(torch.zeros(states_num, states_num, dtype=torch.float))
        nn.init.normal_(self.transition_matrix, -1, 0.1)
        # Default initialization
        self.transition_matrix.data[self.sos_idx, :] = -9999.0
        self.transition_matrix.data[:, self.pad_idx] = -9999.0
        self.transition_matrix.data[self.pad_idx, :] = -9999.0
        self.transition_matrix.data[self.pad_idx, self.pad_idx] = 0.0

    def get_empirical_transition_matrix(self, tag_sequences_train, tag_seq_indexer=None):
        if tag_seq_indexer is None:
            tag_seq_indexer = self.tag_seq_indexer
        empirical_transition_matrix = torch.zeros(self.states_num, self.states_num, dtype=torch.long)
        for tag_seq in tag_sequences_train:
            s = tag_seq_indexer.item2idx_dict[tag_seq[0]]
            empirical_transition_matrix[s, self.sos_idx] += 1
            for n, tag in enumerate(tag_seq):
                if n + 1 >= len(tag_seq):
                    break
                next_tag = tag_seq[n + 1]
                j = tag_seq_indexer.item2idx_dict[tag]
                i = tag_seq_indexer.item2idx_dict[next_tag]
                empirical_transition_matrix[i, j] += 1
        return empirical_transition_matrix

    def init_transition_matrix_empirical(self, tag_sequences_train):
        # Calculate statistics for tag transitions
        empirical_transition_matrix = self.get_empirical_transition_matrix(tag_sequences_train)
        # Initialize
        for i in range(self.tag_seq_indexer.get_items_count()):
            for j in range(self.tag_seq_indexer.get_items_count()):
                if empirical_transition_matrix[i, j] == 0:
                    self.transition_matrix.data[i, j] = -9999.0
                #self.transition_matrix.data[i, j] = torch.log(empirical_transition_matrix[i, j].float() + 10**-32)
        if self.verbose:
            print('Empirical transition matrix from the train dataset:')
            self.pretty_print_transition_matrix(empirical_transition_matrix)
            print('\nInitialized transition matrix:')
            self.pretty_print_transition_matrix(self.transition_matrix.data)

    def pretty_print_transition_matrix(self, transition_matrix, tag_seq_indexer=None):
        if tag_seq_indexer is None:
            tag_seq_indexer = self.tag_seq_indexer
        str = '%10s' % ''
        for i in range(tag_seq_indexer.get_items_count()):
            str += '%10s' % tag_seq_indexer.idx2item_dict[i]
        str += '\n'
        for i in range(tag_seq_indexer.get_items_count()):
            str += '\n%10s' % tag_seq_indexer.idx2item_dict[i]
            for j in range(tag_seq_indexer.get_items_count()):
                str += '%10s' % ('%1.1f' % transition_matrix[i, j])
        print(str)

    def is_cuda(self):
        return self.transition_matrix.is_cuda

    def numerator(self, features_rnn_compressed, states_tensor, mask_tensor):
        # features_input_tensor: batch_num x max_seq_len x states_num
        # states_tensor: batch_num x max_seq_len
        # mask_tensor: batch_num x max_seq_len
        batch_num, max_seq_len = mask_tensor.shape
        score = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
        start_states_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, 1, dtype=torch.long).fill_(self.sos_idx))
        states_tensor = torch.cat([start_states_tensor, states_tensor], 1)
        for n in range(max_seq_len):
            curr_mask = mask_tensor[:, n]
            curr_emission = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
            curr_transition = self.tensor_ensure_gpu(torch.zeros(batch_num, dtype=torch.float))
            for k in range(batch_num):
                curr_emission[k] = features_rnn_compressed[k, n, states_tensor[k, n + 1]].unsqueeze(0)
                curr_states_seq = states_tensor[k]
                curr_transition[k] = self.transition_matrix[curr_states_seq[n + 1], curr_states_seq[n]].unsqueeze(0)
            score = score + curr_emission*curr_mask + curr_transition*curr_mask
        return score

    def denominator(self, features_rnn_compressed, mask_tensor):
        # features_rnn_compressed: batch x max_seq_len x states_num
        # mask_tensor: batch_num x max_seq_len
        batch_num, max_seq_len = mask_tensor.shape
        score = self.tensor_ensure_gpu(torch.zeros(batch_num, self.states_num, dtype=torch.float).fill_(-9999.0))
        score[:, self.sos_idx] = 0.
        for n in range(max_seq_len):
            curr_mask = mask_tensor[:, n].unsqueeze(-1).expand_as(score)
            curr_score = score.unsqueeze(1).expand(-1, *self.transition_matrix.size())
            curr_emission = features_rnn_compressed[:, n].unsqueeze(-1).expand_as(curr_score)
            curr_transition = self.transition_matrix.unsqueeze(0).expand_as(curr_score)
            #curr_score = torch.logsumexp(curr_score + curr_emission + curr_transition, dim=2)
            curr_score = log_sum_exp(curr_score + curr_emission + curr_transition)
            score = curr_score * curr_mask + score * (1 - curr_mask)
        #score = torch.logsumexp(score, dim=1)
        score = log_sum_exp(score)
        return score

    def decode_viterbi(self, features_rnn_compressed, mask_tensor):
        # features_rnn_compressed: batch x max_seq_len x states_num
        # mask_tensor: batch_num x max_seq_len
        batch_size, max_seq_len = mask_tensor.shape
        seq_len_list = [int(mask_tensor[k].sum().item()) for k in range(batch_size)]
        # Step 1. Calculate scores & backpointers
        score = self.tensor_ensure_gpu(torch.Tensor(batch_size, self.states_num).fill_(-9999.))
        score[:, self.sos_idx] = 0.0
        backpointers = self.tensor_ensure_gpu(torch.LongTensor(batch_size, max_seq_len, self.states_num))
        for n in range(max_seq_len):
            curr_emissions = features_rnn_compressed[:, n]
            curr_score = self.tensor_ensure_gpu(torch.Tensor(batch_size, self.states_num))
            curr_backpointers = self.tensor_ensure_gpu(torch.LongTensor(batch_size, self.states_num))
            for curr_state in range(self.states_num):
                T = self.transition_matrix[curr_state, :].unsqueeze(0).expand(batch_size, self.states_num)
                max_values, max_indices = torch.max(score + T, 1)
                curr_score[:, curr_state] = max_values
                curr_backpointers[:, curr_state] = max_indices
            curr_mask = mask_tensor[:, n].unsqueeze(1).expand(batch_size, self.states_num)
            score = score * (1 - curr_mask) + (curr_score + curr_emissions) * curr_mask
            backpointers[:, n, :] = curr_backpointers # shape: batch_size x max_seq_len x state_num
        best_score_batch, last_best_state_batch = torch.max(score, 1)
        # Step 2. Find the best path
        best_path_batch = [[state] for state in last_best_state_batch.tolist()]
        for k in range(batch_size):
            curr_best_state = last_best_state_batch[k]
            curr_seq_len = seq_len_list[k]
            for n in reversed(range(1, curr_seq_len)):
                curr_best_state = backpointers[k, n, curr_best_state].item()
                best_path_batch[k].insert(0, curr_best_state)
        return best_path_batch

class TaggerBase(nn.Module):
    """TaggerBase is an abstract class for tagger models. It implements the tagging functionality for
    different types of inputs (sequences of tokens, sequences of integer indices, tensors). Auxiliary class
    SequencesIndexer is used for input and output data formats conversions. Abstract method `forward` is used in order
    to make these predictions, it have to be implemented in ancestors."""
    def __init__(self,  word_seq_indexer, tag_seq_indexer, gpu, batch_size):
        super(TaggerBase, self).__init__()
        self.word_seq_indexer = word_seq_indexer
        self.tag_seq_indexer = tag_seq_indexer
        self.gpu = gpu
        self.batch_size = batch_size

    def tensor_ensure_gpu(self, tensor):
        if self.gpu >= 0:
            return tensor.cuda(device=self.gpu)
        else:
            return tensor

    def self_ensure_gpu(self):
        if self.gpu >= 0:
            self.cuda(device=self.gpu)
        else:
            self.cpu()

    def save_tagger(self, checkpoint_fn):
        self.cpu()
        torch.save(self, checkpoint_fn)
        self.self_ensure_gpu()

    def forward(self, *input):
        pass

    def predict_idx_from_words(self, word_sequences):
        self.eval()
        outputs_tensor = self.forward(word_sequences) # batch_size x num_class+1 x max_seq_len
        output_idx_sequences = list()
        for k in range(len(word_sequences)):
            idx_seq = list()
            for l in range(len(word_sequences[k])):
                curr_output = outputs_tensor[k, 1:, l] # ignore the first component of output
                max_no = curr_output.argmax(dim=0)
                idx_seq.append(max_no.item() + 1)
            output_idx_sequences.append(idx_seq)
        return output_idx_sequences

    def predict_tags_from_words(self, word_sequences, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        print('\n')
        batch_num = math.floor(len(word_sequences) / batch_size)
        if len(word_sequences) > 0 and len(word_sequences) < batch_size:
            batch_num = 1
        output_tag_sequences = list()
        for n in range(batch_num):
            i = n*batch_size
            if n < batch_num - 1:
                j = (n + 1)*batch_size
            else:
                j = len(word_sequences)
            curr_output_idx = self.predict_idx_from_words(word_sequences[i:j])
            curr_output_tag_sequences = self.tag_seq_indexer.idx2items(curr_output_idx)
            output_tag_sequences.extend(curr_output_tag_sequences)
            print('\r++ predicting, batch %d/%d (%1.2f%%).' % (n + 1, batch_num, math.ceil(n * 100.0 / batch_num)),
                  end='', flush=True)
        return output_tag_sequences

    def get_mask_from_word_sequences(self, word_sequences):
        batch_num = len(word_sequences)
        max_seq_len = max([len(word_seq) for word_seq in word_sequences])
        mask_tensor = self.tensor_ensure_gpu(torch.zeros(batch_num, max_seq_len, dtype=torch.float))
        for k, word_seq in enumerate(word_sequences):
            mask_tensor[k, :len(word_seq)] = 1
        return mask_tensor # batch_size x max_seq_len

    def apply_mask(self, input_tensor, mask_tensor):
        input_tensor = self.tensor_ensure_gpu(input_tensor)
        mask_tensor = self.tensor_ensure_gpu(mask_tensor)
        return input_tensor*mask_tensor.unsqueeze(-1).expand_as(input_tensor)

class TaggerBiRNNCRF(TaggerBase):
    """TaggerBiRNNCRF is a model for sequences tagging that includes recurrent network + CRF."""
    def __init__(self, word_seq_indexer, tag_seq_indexer, class_num, batch_size=1, rnn_hidden_dim=100,
                 freeze_word_embeddings=False, dropout_ratio=0.5, rnn_type='GRU', gpu=-1):
        super(TaggerBiRNNCRF, self).__init__(word_seq_indexer, tag_seq_indexer, gpu, batch_size)
        self.tag_seq_indexer = tag_seq_indexer
        self.class_num = class_num
        self.rnn_hidden_dim = rnn_hidden_dim
        self.freeze_embeddings = freeze_word_embeddings
        self.dropout_ratio = dropout_ratio
        self.rnn_type = rnn_type
        self.gpu = gpu
        self.word_embeddings_layer = LayerWordEmbeddings(word_seq_indexer, gpu, freeze_word_embeddings)
        self.dropout = torch.nn.Dropout(p=dropout_ratio)
        if rnn_type == 'LSTM':
            self.birnn_layer = LayerBiLSTM(input_dim=self.word_embeddings_layer.output_dim,
                                           hidden_dim=rnn_hidden_dim,
                                           gpu=gpu)
        else:
            raise ValueError('Unknown rnn_type = %s, must be either "LSTM" or "GRU"')
        self.lin_layer = nn.Linear(in_features=self.birnn_layer.output_dim, out_features=class_num + 2)
        self.crf_layer = LayerCRF(gpu, states_num=class_num + 2, pad_idx=tag_seq_indexer.pad_idx, sos_idx=class_num + 1,
                                  tag_seq_indexer=tag_seq_indexer)
        if gpu >= 0:
            self.cuda(device=self.gpu)

    def _forward_birnn(self, word_sequences):
        mask = self.get_mask_from_word_sequences(word_sequences)
        z_word_embed = self.word_embeddings_layer(word_sequences)
        z_word_embed_d = self.dropout(z_word_embed)
        rnn_output_h = self.birnn_layer(z_word_embed_d, mask)
        rnn_output_h_d = self.dropout(rnn_output_h) # shape: batch_size x max_seq_len x rnn_hidden_dim*2
        features_rnn_compressed = self.lin_layer(rnn_output_h_d) # shape: batch_size x max_seq_len x class_num
        return self.apply_mask(features_rnn_compressed, mask)

    def get_loss(self, word_sequences_train_batch, tag_sequences_train_batch):
        targets_tensor_train_batch = self.tag_seq_indexer.items2tensor(tag_sequences_train_batch)
        features_rnn = self._forward_birnn(word_sequences_train_batch) # batch_num x max_seq_len x class_num
        mask = self.get_mask_from_word_sequences(word_sequences_train_batch)  # batch_num x max_seq_len
        numerator = self.crf_layer.numerator(features_rnn, targets_tensor_train_batch, mask)
        denominator = self.crf_layer.denominator(features_rnn, mask)
        nll_loss = -torch.mean(numerator - denominator)
        return nll_loss

    def predict_idx_from_words(self, word_sequences):
        self.eval()
        features_rnn_compressed  = self._forward_birnn(word_sequences)
        mask = self.get_mask_from_word_sequences(word_sequences)
        idx_sequences = self.crf_layer.decode_viterbi(features_rnn_compressed, mask)
        return idx_sequences

    def predict_tags_from_words(self, word_sequences, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        print('\n')
        batch_num = math.floor(len(word_sequences) / batch_size)
        if len(word_sequences) > 0 and len(word_sequences) < batch_size:
            batch_num = 1
        output_tag_sequences = list()
        for n in range(batch_num):
            i = n*batch_size
            if n < batch_num - 1:
                j = (n + 1)*batch_size
            else:
                j = len(word_sequences)
            curr_output_idx = self.predict_idx_from_words(word_sequences[i:j])
            curr_output_tag_sequences = self.tag_seq_indexer.idx2items(curr_output_idx)
            output_tag_sequences.extend(curr_output_tag_sequences)
            print('\r++ predicting, batch %d/%d (%1.2f%%).' % (n + 1, batch_num, math.ceil(n * 100.0 / batch_num)),
                  end='', flush=True)
        return output_tag_sequences

class TaggerFactory():
    """TaggerFactory contains wrappers to create various tagger models."""
    @staticmethod
    def load(checkpoint_fn, gpu=-1):
        if not os.path.isfile(checkpoint_fn):
            raise ValueError('Can''t find tagger in file "%s". Please, run the main script with non-empty \
                             "--save-best-path" param to create it.' % checkpoint_fn)
        tagger = torch.load(checkpoint_fn)
        tagger.gpu = gpu

        tagger.word_seq_indexer.gpu = gpu # hotfix
        tagger.tag_seq_indexer.gpu = gpu # hotfix
        if hasattr(tagger, 'char_embeddings_layer'):# very hot hotfix
            tagger.char_embeddings_layer.char_seq_indexer.gpu = gpu # hotfix
        tagger.self_ensure_gpu()
        return tagger

    @staticmethod
    def create(args, word_seq_indexer, tag_seq_indexer, tag_sequences_train):
        tagger = TaggerBiRNNCRF(word_seq_indexer=word_seq_indexer,
                                tag_seq_indexer=tag_seq_indexer,
                                class_num=tag_seq_indexer.get_class_num(),
                                batch_size=args.batch_size,
                                rnn_hidden_dim=args.rnn_hidden_dim,
                                freeze_word_embeddings=args.freeze_word_embeddings,
                                dropout_ratio=args.dropout_ratio,
                                rnn_type=args.rnn_type,
                                gpu=args.gpu)
        tagger.crf_layer.init_transition_matrix_empirical(tag_sequences_train)
        return tagger

class DataIOConnlNer2003():
    """DataIONerConnl2003 is an input/output data wrapper for CoNNL-2003 Shared Task file format.
    Tjong Kim Sang, Erik F., and Fien De Meulder. "Introduction to the CoNLL-2003 shared task: Language-independent
    named entity recognition." Proceedings of the seventh conference on Natural language learning at HLT-NAACL
    2003-Volume 4. Association for Computational Linguistics, 2003.
    """
    def read_train_dev_test(self, args):
        word_sequences_train, tag_sequences_train = self.read_data(fn=args.train, verbose=args.verbose)
        word_sequences_dev, tag_sequences_dev = self.read_data(fn=args.dev, verbose=args.verbose)
        word_sequences_test, tag_sequences_test = self.read_data(fn=args.test, verbose=args.verbose)
        return word_sequences_train, tag_sequences_train, word_sequences_dev, tag_sequences_dev, word_sequences_test, \
               tag_sequences_test

    def read_data(self, fn, verbose=True, column_no=-1):
        word_sequences = list()
        tag_sequences = list()
        with codecs.open(fn, 'r', 'utf-8') as f:
            lines = f.readlines()
        curr_words = list()
        curr_tags = list()
        for k in range(len(lines)):
            line = lines[k].strip()
            if len(line) == 0 or line.startswith('-DOCSTART-'): # new sentence or new document
                if len(curr_words) > 0:
                    word_sequences.append(curr_words)
                    tag_sequences.append(curr_tags)
                    curr_words = list()
                    curr_tags = list()
                continue
            strings = line.split(' ')
            word = strings[0]
            tag = strings[column_no] # be default, we take the last tag
            curr_words.append(word)
            curr_tags.append(tag)
            if k == len(lines) - 1:
                word_sequences.append(curr_words)
                tag_sequences.append(curr_tags)
        if verbose:
            print('Loading from %s: %d samples, %d words.' % (fn, len(word_sequences), get_words_num(word_sequences)))
        return word_sequences, tag_sequences

    def write_data(self, fn, word_sequences, tag_sequences_1, tag_sequences_2):
        text_file = open(fn, mode='w')
        for i, words in enumerate(word_sequences):
            tags_1 = tag_sequences_1[i]
            tags_2 = tag_sequences_2[i]
            for j, word in enumerate(words):
                tag_1 = tags_1[j]
                tag_2 = tags_2[j]
                text_file.write('%s %s %s\n' % (word, tag_1, tag_2))
            text_file.write('\n')
        text_file.close()

