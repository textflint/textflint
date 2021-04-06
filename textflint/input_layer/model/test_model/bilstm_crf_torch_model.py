"""
biLSTM-crf for NER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
from fastNLP import Trainer, BucketSampler, SpanFPreRecMetric
from fastNLP.embeddings import StaticEmbedding
from fastNLP.io import Conll2003NERPipe
from fastNLP.models import BiLSTMCRF
from torch import nn as nn, optim

from textflint.common.utils.load import load_cached_state_dict
from textflint.common import device


def load_data(train_path, dev_path, test_path):
    paths = {'test': test_path,
             'train': train_path,
             'dev': dev_path}
    data = Conll2003NERPipe(encoding_type='bio').process_from_file(paths)
    embed = StaticEmbedding(vocab=data.get_vocab('words'),
                            model_dir_or_name='en-glove-6b-100d',
                            requires_grad=False,
                            lower=True,
                            word_dropout=0,
                            dropout=0.5,
                            only_norm_found_vector=True)
    return data, embed


class BilstmCRFTorchModel(nn.Module):
    r"""A biLSTM-CRF neural network for
    named entity recognition which implemented by pytorch(fastnlp).

    """
    def __init__(
        self,
        init_embedding=None,
        hidden_size=100,
        dropout=0.5,
        tag_vocab=None,
        model_path=None
    ):
        super().__init__()
        self.emb_layer = init_embedding
        self.model = BiLSTMCRF(self.emb_layer,
                               num_classes=len(tag_vocab),
                               hidden_size=hidden_size,
                               dropout=dropout,
                               target_vocab=tag_vocab)
        if model_path is not None:
            self.load_from_disk(model_path)
        self.to(device)

    def load_from_disk(self, model_path):
        self.load_state_dict(load_cached_state_dict(model_path))
        self.to(device)
        self.eval()

    def _forward(self, words, seq_len=None, target=None):
        if target is None:
            pred = self.model(words, seq_len, target)
            return pred
        else:
            loss = self.model(words, seq_len, target)
            return loss

    def forward(self, words, seq_len, target):
        return self._forward(words, seq_len, target)

    def predict(self, words, seq_len):
        return self._forward(words, seq_len)

    def get_input_embeddings(self):
        return self.emb_layer.embedding
