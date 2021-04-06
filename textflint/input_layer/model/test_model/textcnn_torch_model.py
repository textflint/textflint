"""
Word CNN for Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from .glove_embedding import GloveEmbedding
from textflint.common.utils.load import load_cached_state_dict
from textflint.common import device


class TextCNNTorchModel(nn.Module):
    r"""A convolutional neural network for text classification which implemented by pytorch.

    We use different versions of this network to pretrain models for
    text classification.
    """

    def __init__(
        self,
        init_embedding=None,
        embedding_shape=None,
        hidden_size=100,
        dropout=0.2,
        num_labels=2,
        model_path=None
    ):
        super().__init__()
        if init_embedding is not None:
            self.emb_layer = nn.Embedding.from_pretrained(
                torch.from_numpy(init_embedding)
            )
        elif embedding_shape is not None:
            self.emb_layer = nn.Embedding(
                embedding_shape[0],
                embedding_shape[1]
            )
        else:
            raise ValueError("Cant init embedding layer cuz empty embedding "
                             "and empty embedding shape.")
        self.drop = nn.Dropout(dropout)

        self.encoder = CNNTextLayer(
            self.emb_layer.embedding_dim, widths=[3, 4, 5], filters=hidden_size
        )

        d_out = 3 * hidden_size
        self.linear = nn.Linear(d_out, num_labels)
        self.out = nn.Softmax(dim=-1)

        if model_path is not None:
            self.load_from_disk(model_path)

        self.to(device)

    def load_from_disk(self, model_path):
        self.load_state_dict(load_cached_state_dict(model_path))
        self.to(device)
        self.eval()

    def forward(self, _input):
        emb = self.emb_layer(_input)
        emb = self.drop(emb)

        output = self.encoder(emb)

        output = self.drop(output)
        pred = self.linear(output)
        pred = self.out(pred)

        return pred

    def get_input_embeddings(self):
        return self.emb_layer.embedding


class CNNTextLayer(nn.Module):
    def __init__(self, n_in, widths=[3, 4, 5], filters=100):
        super().__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, Ci, len, d)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        return x
