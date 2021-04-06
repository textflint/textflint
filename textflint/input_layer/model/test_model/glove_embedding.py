"""
Glove Embedding
^^^^^^^^^^^^^^^^^^^^^

"""

import os
import numpy as np
import textattack

from ....common.utils import logger, download_if_needed


class Embedding:
    r"""
    A layer of a model that replaces word IDs with their embeddings.

    This is a useful abstraction for any nn.module which wants to take word IDs
    (a sequence of text) as input layer but actually manipulate words'
    embeddings.

    Requires some pre-trained embedding with associated word IDs.

    """

    def __init__(
        self,
        n_d=100,
        embedding_matrix=None,
        word_list=None,
        oov="<oov>",
        pad="<pad>"
    ):
        word2id = {}
        if embedding_matrix is not None:
            for word in word_list:
                assert word not in word2id, \
                    "Duplicate words in pre-trained embeddings"
                word2id[word] = len(word2id)

            logger.debug(f"{len(word2id)} pre-trained "
                         f"word embeddings loaded.\n")

            n_d = len(embedding_matrix[0])

        if oov not in word2id:
            word2id[oov] = len(word2id)

        if pad not in word2id:
            word2id[pad] = len(word2id)

        self.word2id = word2id
        self.vocab_size, self.embedding_size = len(word2id), n_d

        self.embedding = np.random.normal(
            size=(self.vocab_size, self.embedding_size)
        ).astype(np.float32)

        if embedding_matrix is not None:
            self.embedding[:len(embedding_matrix)] \
                = np.array(embedding_matrix).astype(np.float32)[:]

        self.oovid = word2id[oov]
        self.padid = word2id[pad]


class GloveEmbedding(Embedding):
    r"""
    Pre-trained Global Vectors for Word Representation (GLOVE) vectors.
    Uses embeddings of dimension 200.

    GloVe is an unsupervised learning algorithm for obtaining vector
    representations for words. Training is performed on aggregated global
    word-word co-occurrence statistics from a corpus, and the resulting
    representations showcase interesting linear substructures of the word
    vector space.

    GloVe: Global Vectors for Word Representation. (Jeffrey Pennington,
        Richard Socher, and Christopher D. Manning. 2014.)
    """

    EMBEDDING_PATH = "EMBEDDING/glove200.zip"

    def __init__(self):
        glove_path = download_if_needed(
            GloveEmbedding.EMBEDDING_PATH
        )
        # glove_path = download_if_needed(GloveEmbedding.EMBEDDING_PATH)
        glove_word_list_path = os.path.join(glove_path,
                                            "glove200/glove.wordlist.npy")
        word_list = np.load(glove_word_list_path)
        glove_matrix_path = os.path.join(glove_path,
                                         "glove200/glove.6B.200d.mat.npy")
        embedding_matrix = np.load(glove_matrix_path).astype(np.float32)
        super().__init__(embedding_matrix=embedding_matrix, word_list=word_list)
