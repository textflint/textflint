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


if __name__ == "__main__":
    from .model_helper import *
    from ..tokenizers import GloveTokenizer


    def train(training_data, args):
        # device = "GPU:0" if tf.test.is_gpu_available() else "CPU"

        glove_embedding = GloveEmbedding()
        word2id = glove_embedding.word2id
        label2id = {"positive": 0, "negative": 1}

        model = TextCNNTF(init_embedding=glove_embedding.embedding)
        tokenizer = GloveTokenizer(
            word_id_map=word2id,
            unk_token_id=glove_embedding.oovid,
            pad_token_id=glove_embedding.padid,
            max_length=args['max_length']
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=args['lr'])

        steps = 0

        for epoch in range(1, args['epoch'] + 1):
            for inputs, labels in train_iter(training_data, args['batch_size'],
                                             tokenizer, label2id):
                inputs, labels = tf.constant(inputs), torch.from_numpy(labels)

                with tf.GradientTape() as tape:
                    logits = model(inputs)
                    loss = tf.keras.losses.sparse_categorical_crossentropy(
                        y_pred=logits, y_true=labels
                    )
                    loss = tf.reduce_mean(loss)

                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(
                    grads_and_vars=zip(grads, model.variables))

                steps += 1
                if steps % 20 == 0:

                    accuracy = tf.reduce_sum(
                        tf.metrics.sparse_categorical_accuracy(labels, logits))\
                               / args['batch_size']
                    print(f"Step {steps}\t loss {loss}\t acc {accuracy}")

        return model

    # load data
    sa_data_set = data_loader_csv('sa_test.csv')

    print("Finish Load!")
    train_data_set = sa_data_set[:int(len(sa_data_set)*0.7)]
    test_data_set = sa_data_set[int(len(sa_data_set)*0.7):]

    # train stage
    train_args = {'lr': 0.005, 'epoch': 10, 'batch_size': 32, 'max_length': 30}
    model = train(train_data_set, train_args)
    ckpt_dir = './test_model/ckpt/'
    tf_save(model, ckpt_dir, 'wx')
