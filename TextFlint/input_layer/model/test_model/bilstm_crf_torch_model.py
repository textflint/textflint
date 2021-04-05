"""
biLSTM-crf for NER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
from fastNLP import Trainer, BucketSampler, SpanFPreRecMetric
from fastNLP.embeddings import StaticEmbedding
from fastNLP.io import Conll2003NERPipe
from fastNLP.models import BiLSTMCRF
from torch import nn as nn, optim

from TextFlint.common.utils.load import load_cached_state_dict
from TextFlint.common import device


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


if __name__ == "__main__":

    # load data
    ner_train_data_path = r'conll/train.txt'
    ner_dev_data_path = r'conll/dev.txt'
    ner_test_data_set = r'conll/test.txt'
    data_bundle, embedding = load_data(ner_train_data_path,
                                       ner_dev_data_path,
                                       ner_test_data_set)
    # train stage
    bilstm_crf = BilstmCRFTorchModel(init_embedding=embedding,
                                tag_vocab=data_bundle.get_vocab('target'))
    optimizer = optim.SGD(bilstm_crf.parameters(), lr=0.0009, momentum=0.9)
    ckpt_dir = './test_model/ckpt/'
    trainer = Trainer(data_bundle.get_dataset('train'),
                      bilstm_crf,
                      optimizer,
                      batch_size=4,
                      sampler=BucketSampler(),
                      num_workers=0,
                      n_epochs=5,
                      dev_data=data_bundle.get_dataset('dev'),
                      metrics=SpanFPreRecMetric(
                          tag_vocab=data_bundle.get_vocab('target'),
                          encoding_type='bio'),
                      dev_batch_size=4,
                      device=device,
                      test_use_tqdm=False,
                      use_tqdm=True,
                      print_every=300,
                      save_path=ckpt_dir)
    trainer.train(load_best_model=False)
