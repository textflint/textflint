"""
biLSTM-crf for NER
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
import torch
from fastNLP import SpanFPreRecMetric, Tester

from .torch_model import TorchModel
from ..test_model.bilstm_crf_torch_model import BilstmCRFTorchModel, load_data
from ....common import device
from ..test_model.glove_embedding import GloveEmbedding


class BilstmCRFTorch(TorchModel):
    r"""
    Model wrapper for TextCnn implemented by pytorch.

    """
    def __init__(
            self,
            embedding=None,
            tag_vocab=None):

        self.tag_vocab=tag_vocab
        super().__init__(
            model=BilstmCRFTorchModel(
                init_embedding=embedding,
                tag_vocab=self.tag_vocab
            ),
            task='NER',
            tokenizer=None
        )

    def __call__(self, batch_texts):
        r"""
        Tokenize text, convert tokens to id and run the model.

        :param batch_texts: (batch_size,) batch text input
        :return: numpy.array()

        """
        model_device = next(self.model.parameters()).device
        inputs_ids = [self.encode(batch_text) for batch_text in batch_texts]
        ids = torch.tensor(inputs_ids).to(model_device)

        return self.model(ids).detach().cpu().numpy()

    def evaluate(self, data_samples, prefix=''):
        r"""
        :param DataSet data_samples: DataSet with Samples and Vocabs
        :return: dict obj to save metrics result

        """
        # span_f1_metric = SpanFPreRecMetric(
        #     tag_vocab=data_bundle.get_vocab('target'),
        #     encoding_type='bio')
        tester = Tester(data_bundle.get_dataset('test'),
                        self.model,
                        metrics=SpanFPreRecMetric(
                            tag_vocab=data_bundle.get_vocab('target'),
                            encoding_type='bio'),
                        batch_size=4,
                        device=device)

        return tester.test()['SpanFPreRecMetric']



if __name__ == "__main__":
    ner_train_data_path = r'conll/train.txt'
    ner_dev_data_path = r'conll/dev.txt'
    ner_test_data_set = r'conll/test.txt'
    data_bundle, embedding = load_data(ner_train_data_path,
                                       ner_dev_data_path,
                                       ner_test_data_set)
    tag_vocab = data_bundle.get_vocab('target')
    print("finish load!")
    bilstm_crf_wrapper = BilstmCRFTorch(embedding, tag_vocab)

    print(bilstm_crf_wrapper.evaluate(data_bundle))

