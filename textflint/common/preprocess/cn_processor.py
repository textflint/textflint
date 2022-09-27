r"""
CnProcessor Class
============================================
"""
import threading
from ..settings import CN_SYNONYM_PATH, CN_ANTONYM_PATH
from ..utils.install import download_if_needed

__all__ = ['CnProcessor']


class CnProcessor:
    r"""
    Chinese Text Processor class

    """
    _instance_lock = threading.Lock()

    def __init__(self):
        self.__ner = None
        self.__pos = None
        self.__dp = None
        self.__hownet = None
        self.__synonym_dict = None
        self.__antonym_dict = None

    # Single instance mode
    def __new__(cls, *args, **kwargs):
        if not hasattr(CnProcessor, "_instance"):
            with CnProcessor._instance_lock:
                if not hasattr(CnProcessor, "_instance"):
                    CnProcessor._instance = object.__new__(cls)
        return CnProcessor._instance

    @staticmethod
    def tokenize(sent, cws=True):
        r"""
        tokenize fiction

        :param str sent: the sentence need to be tokenized
        :param bool cws: If cws is True, tokenize sentence at word level. If cws is False, tokenize sentence at char level.
        :return: list.the tokens in it
        """
        assert isinstance(sent, str)
        if sent == '':
            return []
        if cws:
            from ltp import LTP
            ltp = LTP()
            segment, _ = ltp.seg([sent])
            return segment[0]
        else:
            return [word for word in sent]

    def get_ner(self, sentence):
        r"""
        NER function.
        :param str sentence: the sentence need to be ner
        :return two forms of tags
            The first is the triple form (tags,start,end)
            The second is the list form, which marks the ner label of each word
            such as 周小明去玩
            ['Nh', 'Nh', 'Nh', 'O', 'O']
        """
        assert isinstance(sentence, (list, str))
        from ltp import LTP
        if isinstance(sentence, list):
            # Turn the list into sentence
            tmp = ''.join([word for word in sentence])
            for word in sentence:
                tmp += word
            sentence = tmp

        if not sentence:
            return [], []

        if self.__ner is None:
            self.__ner = LTP()
        seg, hidden = self.__ner.seg([sentence])
        seg = seg[0]
        ner = self.__ner.ner(hidden)
        ner = ner[0]

        ner_label = len(sentence) * ['O']
        for i in range(len(ner)):
            tag, start, end = ner[i]
            tmp = 0
            for j in range(start):
                tmp += len(seg[j])
            start = tmp
            tmp = 0
            for j in range(end + 1):
                tmp += len(seg[j])
            end = tmp
            ner[i] = (tag, start, end - 1)
            for j in range(start, end):
                ner_label[j] = tag

        return ner, ner_label

    def get_pos_tag(self, sentence):
        r"""
        pos tag function.
        :param str sentence: the sentence need to be ner
        :return: the triple form (tags,start,end)
        """

        assert isinstance(sentence, (list, str))
        from ltp import LTP
        if isinstance(sentence, list):
            # Turn the list into sentence
            tmp = ''
            for word in sentence:
                tmp += word
            sentence = tmp

        if not sentence:
            return []

        if self.__pos is None:
            # get pos tag
            self.__pos = LTP()
        seg, hidden = self.__pos.seg([sentence])
        pos = self.__pos.pos(hidden)
        seg = seg[0]
        pos = pos[0]
        pos_tag = []
        cnt = 0
        for tag in range(len(pos)):
            pos_tag.append([pos[tag], cnt, cnt + len(seg[tag]) - 1])
            cnt += len(seg[tag])

        return pos_tag

    def get_dp(self, sentence):
        r"""
        dependency parsing function.
        :param str sentence: the sentence need to be parsed
        :return: the list of triple form
        """
        assert isinstance(sentence, (list, str))
        from ltp import LTP
        if isinstance(sentence, list):
            # Turn the list into sentence
            tmp = ''
            for word in sentence:
                tmp += word
            sentence = tmp
        if not sentence:
            return []

        if self.__dp is None:
            self.__dp = LTP()
        seg, hidden = self.__dp.seg([sentence])
        dp = self.__dp.dep(hidden)
        return dp[0]

    def sentence_tokenize(self, text):
        r"""
        Split text to sentences.

        :param str text: text string
        :return: list[str]

        """
        assert isinstance(text, str)
        from ltp import LTP
        ltp = LTP()
        sent = ltp.sent_split([text])
        return sent

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
        text = ''.join(tokens)
        return text

    def feature_extract(self, sent):
        r"""
        Generate linguistic tags for tokens.

        :param str sent: input sentence
        :return: list of dict

        """
        sent_pos = self.get_pos_tag(sent)
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
                'ner': word_ner
            }
            tokens.append(word)
        return tokens

    def generate_synonym_dict(self, synonym_file):
        from collections import defaultdict
        synonym_dict = defaultdict(set)
        f = open(synonym_file, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            words = line.strip().split()[1:]
            for i in range(len(words)):
                for j in range(i + 1, len(words)):
                    synonym_dict[words[i]].add(words[j])
                    synonym_dict[words[j]].add(words[i])
        self.__synonym_dict = synonym_dict

    def get_synonym(self, word, n):
        import random
        if self.__synonym_dict is None:
            self.generate_synonym_dict(download_if_needed(CN_SYNONYM_PATH))
        return random.sample(list(self.__synonym_dict[word]), min(n, len(list(self.__synonym_dict[word]))))

    def generate_antonym_dict(self, antonym_file):
        from collections import defaultdict
        antonym_dict = defaultdict(set)
        f = open(antonym_file, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            try:
                word1, word2 = line.strip().replace('-', ' ').replace('—', ' ').replace('─', ' ').replace('―', ' ').split()
            except Exception:
                pass
            antonym_dict[word1].add(word2)
            antonym_dict[word2].add(word1)
        self.__antonym_dict = antonym_dict

    def get_antonym(self, word, n):
        import random
        if self.__antonym_dict is None:
            self.generate_antonym_dict(download_if_needed(CN_ANTONYM_PATH))
        return random.sample(list(self.__antonym_dict[word]), min(n, len(list(self.__antonym_dict[word]))))

    # def get_synonyms(self, sent):
    #     r"""
    #     Get synonyms from Dictionary.
    #
    #     :param str sent: The text.
    #     :return: A list of str, represents the sense of each input token.
    #
    #     """
    #     assert isinstance(sent, str)
    #
    #     def generate_synonym_dict(synonym_file):
    #         from collections import defaultdict
    #         synonym_dict = defaultdict(set)
    #         f = open(synonym_file, 'r', encoding='utf-8')
    #         lines = f.readlines()
    #         for line in lines:
    #             words = line.strip().split()[1:]
    #             for i in range(len(words)):
    #                 for j in range(i + 1, len(words)):
    #                     synonym_dict[words[i]].add(words[j])
    #                     synonym_dict[words[j]].add(words[i])
    #         return synonym_dict
    #
    #     if self.__synonym_dict is None:
    #         self.__synonym_dict = generate_synonym_dict('dictionary/dict_synonym.txt')
    #
    #     words_and_pos = self.get_word_pos(sent, self.get_pos_tag(sent))
    #
    #     ret = []
    #     for word, pos in words_and_pos:
    #         synonyms = []
    #         if self.normalize_pos(pos) is not None:
    #             synonyms = list(self.__synonym_dict[word])
    #         ret.append(synonyms)
    #
    #     return ret
    #
    # def get_antonyms(self, sent):
    #     r"""
    #     Get antonyms from Dictionary.
    #
    #     :param str sent: The text.
    #     :return: A list of str, represents the sense of each input token.
    #
    #     """
    #     assert isinstance(sent, str)
    #
    #     def generate_antonym_dict(antonym_file):
    #         from collections import defaultdict
    #         antonym_dict = defaultdict(set)
    #         f = open(antonym_file, 'r', encoding='utf-8')
    #         lines = f.readlines()
    #         for line in lines:
    #             try:
    #                 word1, word2 = line.strip().replace('-', ' ').replace('—', ' ').replace('─', ' ').replace('―',' ').split()
    #             except Exception:
    #                 pass
    #             antonym_dict[word1].add(word2)
    #             antonym_dict[word2].add(word1)
    #         return antonym_dict
    #
    #     if self.__antonym_dict is None:
    #         self.__antonym_dict = generate_antonym_dict('dictionary/dict_antonym.txt')
    #
    #     words_and_pos = self.get_word_pos(sent, self.get_pos_tag(sent))
    #
    #     ret = []
    #     for word, pos in words_and_pos:
    #         antonyms = []
    #         if self.normalize_pos(pos) is not None:
    #             antonyms = list(self.__antonym_dict[word])
    #         ret.append(antonyms)
    #
    #     return ret

    @staticmethod
    def normalize_pos(pos):
        if pos in ["noun", "verb", "adj"]:
            pp = pos
        else:
            if pos == "n":
                pp = "noun"
            elif pos == "v":
                pp = "verb"
            elif pos == "a":
                pp = "adj"
            else:
                pp = None

        return pp

    @staticmethod
    def get_word_pos(sent, pos):
        ret = []
        for word in pos:
            start = word[1]
            end = word[2]
            pp = word[0]
            ret.append([sent[start:end+1], pp])
        return ret


# if __name__ == '__main__':
#     sent = '小明想去吃螺蛳粉。'
#     processor = CnProcessor()
#     words = processor.tokenize(sent)
#     pos = processor.get_pos_tag(sent)
#     dp = processor.get_dp(sent)
#     ner = processor.get_ner(sent)
#     syn = processor.get_synonyms(sent)
#     ant = processor.get_antonyms(sent)
#     print(words)
#     print(pos)
#     print(ner)
#     print(dp)
#     print(syn)
#     print(ant)