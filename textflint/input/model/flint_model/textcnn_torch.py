import torch

from .torch_model import TorchModel
from ..test_model.textcnn_torch_model import TextCNNTorchModel
from ..test_model.glove_embedding import GloveEmbedding
from ..tokenizers.glove_tokenizer import GloveTokenizer


class TextCNNTorch(TorchModel):
    r"""
    Model wrapper for TextCnn implemented by pytorch.

    """
    def __init__(self):
        glove_embedding = GloveEmbedding()
        word2id = glove_embedding.word2id

        super().__init__(
            model=TextCNNTorchModel(
                init_embedding=glove_embedding.embedding
            ),
            task='SA',
            tokenizer=GloveTokenizer(
                word_id_map=word2id,
                unk_token_id=glove_embedding.oovid,
                pad_token_id=glove_embedding.padid,
                max_length=30
            )
        )
        self.label2id = {"positive": 0, "negative": 1}

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

    def encode(self, inputs):
        r"""
        Tokenize inputs and convert it to ids.

        :param inputs: model original input
        :return: list of inputs ids

        """
        return self.tokenizer.encode(inputs)

    def unzip_samples(self, data_samples):
        r"""
        Unzip sample to input texts and labels.

        :param list[Sample] data_samples: list of Samples
        :return: (inputs_text), labels.

        """
        x = []
        y = []

        for sample in data_samples:
            x.append(sample['x'])
            y.append(self.label2id[sample['y']])

        return [x], y

