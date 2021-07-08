"""
biLSTM-crf for NER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
import codecs
import math
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def log_sum_exp(x):
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))

def get_words_num(word_sequences):
    return sum(len(word_seq) for word_seq in word_sequences)

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

