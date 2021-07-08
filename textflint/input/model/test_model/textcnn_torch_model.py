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


if __name__ == "__main__":
    from .model_helper import *
    from ..tokenizers import GloveTokenizer

    glove_embedding = GloveEmbedding()
    word2id = glove_embedding.word2id
    label2id = {"positive": 0, "negative": 1}

    tokenizer = GloveTokenizer(
        word_id_map=word2id,
        unk_token_id=glove_embedding.oovid,
        pad_token_id=glove_embedding.padid,
        max_length=30
    )


    def train(training_data, args):


        model = TextCNNTorchModel(init_embedding=glove_embedding.embedding)

        optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
        steps = 0

        for epoch in range(1, args['epoch'] + 1):
            for inputs, labels in train_iter(training_data, args['batch_size'], tokenizer, label2id):
                inputs, labels = torch.from_numpy(inputs).to(device), torch.from_numpy(labels).to(device)
                optimizer.zero_grad()
                logits = model(inputs)

                loss = F.cross_entropy(logits, labels)/args['batch_size']
                loss.backward()
                optimizer.step()

                steps += 1
                if steps % 20 == 0:
                    result = torch.max(logits, 1)[1].view(labels.size())
                    corrects = (result.data == labels.data).sum()
                    accuracy = corrects * 100.0 / args['batch_size']
                    print(f"Step {steps}\t loss {loss}\t acc {accuracy}")

        return model

    def eval(model, test_data):
        count = 0
        acc_count = 0
        model.eval()

        for inputs, labels in train_iter(test_data, 1,
                                         tokenizer, label2id):
            inputs, labels = torch.from_numpy(inputs).to(
                device), torch.from_numpy(labels).to(device)
            logits = model(inputs)
            result = torch.max(logits, 1)[1].view(labels.size())
            count += 1
            acc_count += (result.data == labels.data).sum()

        accuracy = acc_count.item() / count * 100.0
        print(f"acc {accuracy}")

    # load data
    sa_data_set = data_loader_csv('/Users/wangxiao/Desktop/demo/sa_test.csv')

    train_data_set = sa_data_set[:int(len(sa_data_set)*0.7)]
    test_data_set = sa_data_set[int(len(sa_data_set)*0.7):]

    # train stage
    train_args = {'lr': 0.0005, 'epoch': 20, 'batch_size': 32, 'max_length': 30}
    model = train(train_data_set, train_args)
    ckpt_file = '/Users/wangxiao/Desktop/demo/ckpt/test.pkl'
    torch.save(model.state_dict(), open(ckpt_file, "wb"))

    # evaluate stage
    eval(model, test_data_set)
    # model = model_wrapper.model
    loaded_model = TextCNNTorchModel(init_embedding=glove_embedding.embedding)
    loaded_model.load_state_dict(
        torch.load(open(ckpt_file, "rb"))
    )
    eval(loaded_model, test_data_set)


