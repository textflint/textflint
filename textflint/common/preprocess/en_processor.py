"""
EnProcessor Class
============================================

"""

import re
import nltk
import spacy
import threading
from functools import reduce
from spacy.tokens import Doc

from .nltk_res_load import *
from ..settings import *


class EnProcessor:
    r"""
    Text Processor class implement NER, POS tag, lexical tree parsing.
    EnProcessor is designed by single instance mode.

    """
    _instance_lock = threading.Lock()
    initialized = False
    nlp = None
    model_manager = None

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

    # Single instance mode
    def __new__(cls, *args, **kwargs):
        if not hasattr(EnProcessor, "_instance"):
            with EnProcessor._instance_lock:
                if not hasattr(EnProcessor, "_instance"):
                    EnProcessor._instance = object.__new__(cls)
        return EnProcessor._instance

    @classmethod
    def check_initialized(cls):
        if not cls.initialized:
            cls.load_resource()

    @classmethod
    def load_resource(cls):
        if cls.model_manager is None:
            cls.model_manager = ModelManager()
        if cls.nlp is None:
            cls.nlp = spacy.load(
                download_if_needed(MODEL_PATH_WEB) + MODEL_PATH
            )
        cls.initialized = True

    def sentence_tokenize(self, text):
        r"""
        Split text to sentences.

        :param str text: text string
        :return: list[str]

        """
        assert isinstance(text, str)
        self.check_initialized()
        text = self.nlp.tokenizer(text)

        if not self.__sent_tokenizer:
            if 'sentencizer' not in [
                pipeline[0] for pipeline in self.nlp.pipeline
            ]:
                self.nlp.add_pipe('sentencizer')
            self.__sent_tokenizer = self.nlp.pipeline[-1][1]

        return [sent.text for sent in self.__sent_tokenizer(text).sents]

    def tokenize_one_sent(self, text, split_by_space=False):
        r"""
        Tokenize one sentence.

        :param str text:
        :param bool split_by_space: whether tokenize sentence by split space
        :return: tokens

        """
        assert isinstance(text, str)

        if split_by_space:
            return text.split(" ")
        else:
            self.check_initialized()
            return [
                word.text.replace("''", '"')
                    .replace("``", '"') for word in self.nlp.tokenizer(text)
                if word.text != ' ' * len(word)
            ]

    def tokenize(self, text, is_one_sent=False, split_by_space=False):
        """
        Split a text into tokens (words, morphemes we can separate such as
        "n't", and punctuation).

        :param str text:
        :param bool is_one_sent:
        :param bool split_by_space:
        :return: list of tokens

        """
        assert isinstance(text, str)

        def _tokenize_gen(text):
            if is_one_sent:
                yield self.tokenize_one_sent(
                    text,
                    split_by_space=split_by_space
                )
            else:
                for sent in self.sentence_tokenize(text):
                    yield self.tokenize_one_sent(
                        sent,
                        split_by_space=split_by_space
                    )

        return reduce(lambda x, y: x + y, list(_tokenize_gen(text)), [])

    @staticmethod
    def inverse_tokenize(tokens):
        r"""
        Convert tokens to sentence.

        Untokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `untokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.

        Watch out!
        Default punctuation add to the word before its index,
        it may raise inconsistency bug.

        :param list[str]r tokens: target token list
        :return: str

        """
        assert isinstance(tokens, list)
        text = ' '.join(tokens)
        step1 = text.replace("`` ", '"') \
            .replace(" ''", '"') \
            .replace('. . .', '...')
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
        step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
            "can not", "cannot")
        step6 = step5.replace(" ` ", " '")
        step7 = step6.replace('do nt', 'dont').replace('Do nt', 'Dont')
        step8 = step7.replace(' - ', '-')
        return step8.strip()

    def get_pos(self, sentence):
        r"""
        POS tagging function.

        Example::

            EnProcessor().get_pos(
                'All things in their being are good for something.'
            )

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

        :param str|list sentence: A sentence which needs to be tokenized.
        :return: Tokenized tokens with their POS tags.

        """
        assert isinstance(sentence, (str, list))
        self.check_initialized()

        tokens = self.tokenize(sentence) if isinstance(
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
        r"""
        NER function.
        This method uses implemented based on spacy model.

        Example::

            EnProcessor().get_ner(
                'Lionel Messi is a football player from Argentina.'
            )

            if return_word_index is False
            >>[('Lionel Messi', 0, 12, 'PERSON'),
               ('Argentina', 39, 48, 'LOCATION')]

            if return_word_index is True
            >>[('Lionel Messi', 0, 2, 'PERSON'),
               ('Argentina', 7, 8, 'LOCATION')]

        :param str|list sentence: text string or token list
        :param bool return_char_idx: if set True, return character start to
         end index, else return char start to end index.
        :return: A list of tuples, *(entity, start, end, label)*

        """
        self.check_initialized()

        if self.__ner is None:
            self.__ner = self.nlp.pipeline[3][1]

        if self.__word2vec is None:
            self.__word2vec = self.nlp.pipeline[0][1]

        if isinstance(sentence, list):
            tokens = sentence
        elif isinstance(sentence, str):
            tokens = self.tokenize(sentence)  # list of tokens
        else:
            raise ValueError(
                'Support string or token list input, '
                'while your input type is {0}'.format(type(sentence))
            )

        tokens = self.__word2vec(Doc(self.nlp.vocab, words=tokens))
        ner = self.__ner(tokens)

        if return_char_idx is True:
            return [(ent.text, ent.start_char, ent.end_char,
                     self._change_label(ent.label_)) for ent in ner.ents]

        return [(ent.text, ent.start, ent.end, self._change_label(ent.label_))
                for ent in ner.ents]

    def get_parser(self, sentence):
        r"""
        Lexical tree parsing function based on NLTK toolkit.

        Example::

            EnProcessor().get_parser('Messi is a football player.')

            >>'(ROOT\n  (S\n    (NP (NNP Messi))\n    (VP (VBZ is) (NP (DT a)
            (NN football) (NN player)))\n    (. .)))'


        :param str|list sentence: A sentence needs to be parsed.
        :return:The result tree of lexicalized parser in string format.

        """
        self.check_initialized()

        if self.__parser is None:
            self.__parser = self.model_manager.load(CFG_PARSER)

        if not isinstance(sentence, (str, list)):
            raise ValueError('Support string or token list input, while your '
                             'input type is {0}'.format(type(sentence)))
        elif sentence in ['', []]:
            return ''

        sentence = self.inverse_tokenize(sentence) \
            if isinstance(sentence, list) else sentence  # concatenate tokens

        return str(list(self.__parser(sentence))[0])

    def get_dep_parser(self, sentence, is_one_sent=True, split_by_space=False):
        r"""
        Dependency parsing based on spacy model.

        Example::

            EnProcessor().get_dep_parser(
            'The quick brown fox jumps over the lazy dog.'
            )

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

        :param str|list sentence: input text string
        :param bool is_one_sent: whether do sentence tokenzie
        :param bool split_by_space: whether tokenize sentence by split with " "
        :return: dp tags.

        """
        self.check_initialized()

        if self.__dp_parser is None:
            self.__dp_parser = self.nlp.pipeline[2][1]

        if self.__word2vec is None:
            self.__word2vec = self.nlp.pipeline[0][1]

        if self.__pos_tag is None:
            self.__pos_tag = self.nlp.pipeline[1][1]

        if isinstance(sentence, list):
            tokens = sentence
        elif isinstance(sentence, str):
            # list of tokens
            tokens = self.tokenize(sentence, is_one_sent, split_by_space)
        else:
            raise ValueError(
                'Support string or token list input, '
                'while your input type is {0}'.format(type(sentence))
            )

        tokens = self.__pos_tag(
            self.__word2vec(Doc(self.nlp.vocab, words=tokens))
        )
        parse = self.__dp_parser(tokens)

        return [
            (word.text, word.tag_,
             word.head.i + 1 if word.dep_ != 'ROOT' else 0, word.dep_)
            for word in parse
        ]

    def get_lemmas(self, token_and_pos):
        r"""
        Lemmatize function.
        This method uses ``nltk.WordNetLemmatier`` to lemmatize tokens.

        :param list token_and_pos: *(token, POS)*.
        :return: A lemma or a list of lemmas depends on your input.

        """
        self.check_initialized()

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
        r"""
        Lemmatize function for all words in WordNet.

        :param pos: POS tag pr a list of POS tag.
        :return: A list of lemmas that have the given pos tag.

        """
        self.check_initialized()

        if self.__wordnet is None:
            self.__wordnet = self.model_manager.load(NLTK_WORDNET)

        if not isinstance(pos, list):
            return self.__wordnet.all_lemma(pos)
        else:
            return [self.__wordnet.all_lemma(_pos) for _pos in pos]

    def get_delemmas(self, lemma_and_pos):
        r"""
        Delemmatize function.

        This method uses a pre-processed dict which maps (lemma, pos) to
        original token for delemmatizing.

        :param tuple|list lemma_and_pos: A tuple or a list of *(lemma, POS)*.
        :return: A word or a list of words, each word represents the specific
            form of input lemma.

        """
        self.check_initialized()

        if self.__delemmatize is None:
            self.__delemmatize = self.model_manager.load(NLTK_WORDNET_DELEMMA)
        if not isinstance(lemma_and_pos, list):
            token, pos = lemma_and_pos
            return (
                self.__delemmatize[token][pos] if (
                    token in self.__delemmatize) and (
                    pos in self.__delemmatize[token])
                else token
            )
        else:
            return [
                self.__delemmatize[token][pos]
                if (token in self.__delemmatize) and
                (pos in self.__delemmatize[token])
                else token for token, pos in lemma_and_pos
            ]

    def get_synsets(self, tokens_and_pos, lang="eng"):
        r"""
        Get synsets from WordNet.

        :param list tokens_and_pos: A list of tuples, *(token, POS)*.
        :param str lang: language name
        :return: A list of str, represents the sense of each input token.

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
        r"""
        Get antonyms from WordNet.

        This method uses NTLK WordNet to generate antonyms, and uses "lesk"
        algorithm which is proposed by Michael E. Lesk in 1986, to screen
        the sense out.


        :param list tokens_and_pos: A list of tuples, *(token, POS)*.
        :param str lang: language name.
        :return: A list of str, represents the sense of each input token.

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

    def filter_candidates_by_pos(self, token_and_pos, candidates):
        r"""
        Filter synonyms not contain the same pos tag with given token.

        :param list|tuple token_and_pos: *(token, pos)*
        :param list candidates: strings to verify
        :return: filtered candidates list.

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
        r"""
        Generate linguistic tags for tokens.

        :param str sent: input sentence
        :return: list of dict

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
            word = {
                'word': text,
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
