"""
EnProcessor Class
============================================

"""

import nltk
import spacy
import threading
from spacy.tokens import Doc

from .nltk_res_load import *
from ..settings import *
from .tokenizer import tokenize, untokenize, sentence_tokenize


class EnProcessor:

    """ Text Processor class implement NER, POS tag, lexical tree parsing with ``nltk`` toolkit.

    EnProcessor is designed by single instance mode.

    """
    _instance_lock = threading.Lock()
    nlp = spacy.load(download_if_needed(MODEL_PATH_WEB) + MODEL_PATH)

    def __init__(self):
        self.nltk = __import__("nltk")
        self.__sent_tokenizer = None
        self.__lemmatize = None
        self.__delemmatize = None
        self.__ner = None
        self.__pos_tag = None
        self.__parser = None
        self.__dp_parser = None
        self.__wordnet = None
        self.__word2vec = None
        self.__attribute_ruler = None
        self.model_manager = ModelManager()

    # Single instance mode
    def __new__(cls, *args, **kwargs):
        if not hasattr(EnProcessor, "_instance"):
            with EnProcessor._instance_lock:
                if not hasattr(EnProcessor, "_instance"):
                    EnProcessor._instance = object.__new__(cls)
        return EnProcessor._instance

    @staticmethod
    def word_tokenize(sent, is_one_sent=False, split_by_space=False):
        """ Split sentences and tokenize.

        Args:
            sent: target string
            is_one_sent: whether split sentence
            split_by_space: whether split bu space or tokenizer

        Returns:
            tokens

        """
        assert isinstance(sent, str)
        return tokenize(sent, is_one_sent=is_one_sent,
                        split_by_space=split_by_space)

    @staticmethod
    def inverse_tokenize(tokens):
        assert isinstance(tokens, list)
        return untokenize(tokens)

    def sentence_tokenize(self, paras):
        assert isinstance(paras, str)
        if self.__sent_tokenizer is None:
            self.__sent_tokenizer = sentence_tokenize

        return self.__sent_tokenizer(paras)

    def get_pos(self, sentence):
        """ POS tagging function.

        Example:
            EnProcessor().get_pos('All things in their being are good for something.')

            >> [('All', 'DT'),
                ('things', 'NNS'),
                ('in', 'IN'),
                ('their', 'PRP$'),
                ('being', 'VBG'),
                ('are', 'VBP'),
                ('good', 'JJ'),
                ('for', 'IN'),
                ('something', 'NN'),
                ('.', '.')]

        Args:
            sentence: str or list.
                A sentence which needs to be tokenized.

        Returns:
            Tokenized tokens with their POS tags.

        """
        assert isinstance(sentence, (str, list))
        tokens = tokenize(sentence) if isinstance(
            sentence, str) else sentence  # concatenate tokens

        if self.__word2vec is None:
            self.__word2vec = self.nlp.pipeline[0][1]

        tokens = self.__word2vec(Doc(self.nlp.vocab, words=tokens))

        if self.__pos_tag is None:
            self.__pos_tag = self.nlp.pipeline[1][1]

        pos_tags = self.__pos_tag(tokens)

        return [(word.text, word.tag_) for word in pos_tags]

    @staticmethod
    def _change_label(label):
        if label == 'GPE' or label == 'LOC':
            return 'LOCATION'
        if label == 'ORG':
            return 'ORGANIZATION'

        return label

    def get_ner(self, sentence, return_char_idx=True):
        """ NER function.

        This method uses spacy tokenizer and Stanford NER toolkit which requires Java installed.

        Example:
            EnProcessor().get_ner('Lionel Messi is a football player from Argentina.')

            if return_word_index is False
            >>[('Lionel Messi', 0, 12, 'PERSON'),
               ('Argentina', 39, 48, 'LOCATION')]

            if return_word_index is True
            >>[('Lionel Messi', 0, 2, 'PERSON'),
               ('Argentina', 7, 8, 'LOCATION')]

        Args:
            sentence: str or list
                A sentence that we want to extract named entities.
            return_char_idx: bool
                if set True, return character start to end index, else return char start to end index.

        Returns:
            A list of tuples, *(entity, start, end, label)*

        """
        if self.__ner is None:
            self.__ner = self.nlp.pipeline[3][1]

        if self.__word2vec is None:
            self.__word2vec = self.nlp.pipeline[0][1]

        if isinstance(sentence, list):
            tokens = sentence
        elif isinstance(sentence, str):
            tokens = self.word_tokenize(sentence)  # list of tokens
        else:
            raise ValueError(
                'Support string or token list input, while your input type is {0}'.format(
                    type(sentence)))

        tokens = self.__word2vec(Doc(self.nlp.vocab, words=tokens))
        ner = self.__ner(tokens)

        if return_char_idx is True:
            return [(ent.text, ent.start_char, ent.end_char,
                     self._change_label(ent.label_)) for ent in ner.ents]

        return [(ent.text, ent.start, ent.end, self._change_label(ent.label_))
                for ent in ner.ents]

    def get_parser(self, sentence):
        """ Lexical tree parsing.

        This method uses Stanford LexParser to generate a lexical tree which requires Java installed.

        Example:
            EnProcessor().get_parser('Messi is a football player.')

            >>'(ROOT\n  (S\n    (NP (NNP Messi))\n    (VP (VBZ is) (NP (DT a) (NN football) (NN player)))\n    (. .)))'

        Args:
            sentence: str or list.
                A sentence needs to be parsed.

        Returns:
            The result tree of lexicalized parser in string format.

        """

        if self.__parser is None:
            self.__parser = self.model_manager.load(CFG_PARSER)

        if not isinstance(sentence, (str, list)):
            raise ValueError('Support string or token list input, while your input type is {0}'.format(
                type(sentence)))
        elif sentence in ['', []]:
            return ''

        sentence = untokenize(sentence) if isinstance(
            sentence, list) else sentence  # concatenate tokens

        return str(list(self.__parser(sentence))[0])

    def get_dep_parser(self, sentence, is_one_sent=True, split_by_space=False):
        """ Dependency parsing.

        Example:
            EnProcessor().get_dep_parser('The quick brown fox jumps over the lazy dog.')

            >>
                The	DT	4	det
                quick	JJ	4	amod
                brown	JJ	4	amod
                fox	NN	5	nsubj
                jumps	VBZ	0	root
                over	IN	9	case
                the	DT	9	det
                lazy	JJ	9	amod
                dog	NN	5	obl

        Args:
            sentence: str or list.
                A sentence needs to be parsed.
            is_one_sent: bool
            split_by_space: bool

        Returns:

        """
        if self.__dp_parser is None:
            self.__dp_parser = self.nlp.pipeline[2][1]

        if self.__word2vec is None:
            self.__word2vec = self.nlp.pipeline[0][1]

        if self.__pos_tag is None:
            self.__pos_tag = self.nlp.pipeline[1][1]

        if isinstance(sentence, list):
            tokens = sentence
        elif isinstance(sentence, str):
            tokens = self.word_tokenize(sentence, is_one_sent, split_by_space)  # list of tokens
        else:
            raise ValueError('Support string or token list input, while your input type is {0}'.format(
                type(sentence)))

        tokens = self.__pos_tag(self.__word2vec(Doc(self.nlp.vocab, words=tokens)))
        parse = self.__dp_parser(tokens)

        return [(word.text, word.tag_, word.head.i + 1 if word.dep_ != 'ROOT' else 0, word.dep_) for word in parse]

    def get_lemmas(self, token_and_pos):
        """ Lemmatize function.

        This method uses ``nltk.WordNetLemmatier`` to lemmatize tokens.

        Args:
            token_and_pos: list,  *(token, POS)*.

        Returns:
            A lemma or a list of lemmas depends on your input.

        """
        if not isinstance(token_and_pos, list):
            token_and_pos = [token_and_pos]
        if self.__lemmatize is None:
            self.__lemmatize = self.nlp.pipeline[5][1]

        if self.__word2vec is None:
            self.__word2vec = self.nlp.pipeline[0][1]

        if self.__pos_tag is None:
            self.__pos_tag = self.nlp.pipeline[1][1]

        if self.__attribute_ruler is None:
            self.__attribute_ruler = self.nlp.pipeline[4][1]

        tokens = [tp[0] for tp in token_and_pos]
        tokens = Doc(self.nlp.vocab, words=tokens)
        tokens = self.__lemmatize(
            self.__attribute_ruler(
                self.__pos_tag(
                    self.__word2vec(tokens))))

        return [token.lemma_ for token in tokens]

    def get_all_lemmas(self, pos):
        """ Lemmatize function for all words in WordNet.

        This method uses ``nltk.WordNetLemmatier`` to lemmatize tokens.

        Args:
            pos: POS tag pr a list of POS tag.

        Returns:
            A list of lemmas that have the given pos tag.

        """
        if self.__lemmatize is None:
            self.__lemmatize = self.model_manager.load(NLTK_WORDNET).all_lemma

        if not isinstance(pos, list):
            return self.__lemmatize(pos)
        else:
            return [self.__lemmatize(_pos) for _pos in pos]

    def get_delemmas(self, lemma_and_pos):
        """ Delemmatize function.

        This method uses a pre-processed dict which maps (lemma, pos) to original token for delemmatizing.

        Args:
            lemma_and_pos: list or tuple.
                A tuple or a list of tuples, *(lemma, POS)*.

        Returns:
            A word or a list of words, each word represents the specific form of input lemma.

        """

        if self.__delemmatize is None:
            self.__delemmatize = self.model_manager.load(NLTK_WORDNET_DELEMMA)
        if not isinstance(lemma_and_pos, list):
            token, pos = lemma_and_pos
            return (
                self.__delemmatize[token][pos] if (
                    token in self.__delemmatize) and (
                    pos in self.__delemmatize[token]) else token)
        else:
            return [
                self.__delemmatize[token][pos]
                if (token in self.__delemmatize) and (pos in self.__delemmatize[token])
                else token
                for token, pos in lemma_and_pos
            ]

    def get_synsets(self, tokens_and_pos, lang="eng"):
        """ Get synsets from WordNet.

        This method uses NTLK WordNet to generate synsets, and uses "lesk" algorithm which
        is proposed by Michael E. Lesk in 1986, to screen the sense out.

        Args:
            tokens_and_pos: A list of tuples, *(token, POS)*.

        Returns:
            A list of str, represents the sense of each input token.
        """
        if self.__wordnet is None:
            self.__wordnet = self.model_manager.load(NLTK_WORDNET)

        if isinstance(tokens_and_pos, str):
            tokens_and_pos = self.get_pos(tokens_and_pos)

        def lesk(sentence, word, pos):
            synsets = self.__wordnet.synsets(word, lang=lang)

            if pos is not None:
                synsets = [ss for ss in synsets if str(ss.pos()) == pos]

            return synsets

        sentoken = []
        sentence = []
        # normalize pos tag
        for word, pos in tokens_and_pos:
            sentoken.append(word)
            sentence.append((word, self.normalize_pos(pos)))
        ret = []

        for word, pos in sentence:
            ret.append(lesk(sentoken, word, pos))
        return ret

    def get_antonyms(self, tokens_and_pos, lang="eng"):
        """ Get antonyms from WordNet.

        This method uses NTLK WordNet to generate antonyms, and uses "lesk" algorithm which
        is proposed by Michael E. Lesk in 1986, to screen the sense out.

        Args:
            tokens_and_pos: A list of tuples, *(token, POS)*.

        Returns:
            A list of str, represents the sense of each input token.
        """
        if self.__wordnet is None:
            self.__wordnet = self.model_manager.load(NLTK_WORDNET)

        if isinstance(tokens_and_pos, str):
            tokens_and_pos = self.get_pos(tokens_and_pos)

        def lesk(sentence, word, pos):
            synsets = self.__wordnet.synsets(word, lang=lang)
            antonyms = set()

            for synonym in synsets:
                for l in synonym.lemmas():
                    if l.antonyms():
                        antonyms.add(l.antonyms()[0].synset())

            if pos is not None:
                antonyms = [ss for ss in antonyms if str(ss.pos()) == pos]

            return antonyms

        sentoken = []
        sentence = []
        # normalize pos tag
        for word, pos in tokens_and_pos:
            sentoken.append(word)
            sentence.append((word, self.normalize_pos(pos)))
        ret = []

        for word, pos in sentence:
            ret.append(lesk(sentoken, word, pos))
        return ret

    def filter_candidates_by_pos(self, token_and_pos, candidates, lang="eng"):
        """ Filter synonyms not contain the same pos tag with given token.

        Args:
            token_and_pos: list/tuple
            candidates: list

        Returns:
            filtered candidates list.
        """
        def lesk(word, pos, candidates):
            can_word_pos = []

            for candidate in candidates:
                can_word, can_pos = nltk.tag.pos_tag([candidate])[0]
                can_word_pos.append([can_word, self.normalize_pos(can_pos)])

            if pos is not None:
                return [ss[0] for ss in can_word_pos if str(ss[1]) == pos]
            else:
                return []

        # normalize pos tag
        word, pos = token_and_pos

        return lesk(word, self.normalize_pos(pos), candidates)

    def feature_extract(self, sent):
        """Generate linguistic tags for tokens
        Args:
            sent: str

        Returns: list of dict

        """
        sent_pos = self.get_pos(sent)
        sent_lemma = self.get_lemmas(sent_pos)
        ner = self.get_ner(sent)
        tokens = []
        ner_num = len(ner)
        ner_idx = 0
        it, ner_start, ner_end, ner_type = 0, None, None, None
        if ner_num > 0:
            _, ner_start, ner_end, ner_type = ner[ner_idx]
        for i, tok in enumerate(sent_pos):
            text, pos = tok
            it += sent[it:].find(text)
            if ner_num == 0:
                word_ner = 'O'
            else:
                if it > ner_end and ner_idx <= ner_num - 1:
                    ner_idx += 1
                    if ner_idx < ner_num:
                        _, ner_start, ner_end, ner_type = ner[ner_idx]

                if ner_idx == ner_num:
                    word_ner = "O"
                elif ner_start <= it < ner_end:
                    word_ner = ner_type
                else:
                    word_ner = "O"
            word = {'word': text,
                    'pos': pos,
                    'lemma': sent_lemma[i],
                    'ner': word_ner
                    }
            tokens.append(word)
        return tokens

    @staticmethod
    def normalize_pos(pos):
        if pos in ["a", "r", "n", "v", "s"]:
            pp = pos
        else:
            if pos[:2] == "JJ":
                pp = "a"
            elif pos[:2] == "VB":
                pp = "v"
            elif pos[:2] == "NN":
                pp = "n"
            elif pos[:2] == "RB":
                pp = "r"
            else:
                pp = None

        return pp
