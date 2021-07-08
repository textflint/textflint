import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, Dense, \
    MaxPooling1D, Dropout
from tensorflow.keras import Model

from .glove_embedding import GloveEmbedding


class TextCNNTF(Model):
    r"""
    A convolutional neural network for text classification which
    implemented by tensorflow.

    We use different versions of this network to pretrain models for
    text classification.

    """

    def __init__(
        self,
        init_embedding=None,
        hidden_size=100,
        dropout=0.2,
        num_labels=2,
        seq_length=30
    ):

        super(TextCNNTF, self).__init__()
        self.num_labels = num_labels

        glove_embedding = GloveEmbedding()
        self.emb_layer = Embedding(input_dim=glove_embedding.vocab_size,
                                   output_dim=glove_embedding.embedding_size,
                                   weights=[init_embedding])
        self.drop = Dropout(dropout)
        self.encoder = CNNTextLayer(
            seq_length=seq_length,
            widths=[3, 4, 5],
            filters=hidden_size
        )

        self.classifier = Dense(num_labels, activation='softmax')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        if len(inputs.get_shape()) != 2:
            raise ValueError(f'The rank of inputs of TextCNN must be 2, '
                             f'but now is {inputs.get_shape()}')

        emb = self.emb_layer(inputs)
        emb = self.drop(emb)

        output = self.encoder(emb)
        output = tf.squeeze(output)

        # (batch_size, len(self.kernel_sizes)*filters)
        output = self.classifier(output)

        return output


class CNNTextLayer(Model):
    def __init__(
        self,
        seq_length=30,
        widths=[3, 4, 5],
        filters=100
    ):
        super().__init__()
        self.conv1s = [Conv1D(filters=filters, kernel_size=w, padding="same")
                       for w in widths]
        self.pools = [MaxPooling1D(pool_size=seq_length) for w in widths]

    def call(self, x):
        x = [tf.nn.relu(conv(x)) for conv in self.conv1s]
        x = [self.pools[i](x[i]) for i in range(len(x))]
        x = tf.concat(x, axis=-1)
        return x
