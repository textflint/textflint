r"""
CnProcessor Class
============================================
"""
import threading

__all__ = ['CnProcessor']


class CnProcessor:
    r"""
    Text Processor class implement NER.

    """
    _instance_lock = threading.Lock()

    def __init__(self):
        self.__ner = None
        self.__pos = None

    # Single instance mode
    def __new__(cls, *args, **kwargs):
        if not hasattr(CnProcessor, "_instance"):
            with CnProcessor._instance_lock:
                if not hasattr(CnProcessor, "_instance"):
                    CnProcessor._instance = object.__new__(cls)
        return CnProcessor._instance

    @staticmethod
    def tokenize(sent):
        r"""
        tokenize fiction

        :param str sent: the sentence need to be tokenized
        :return: list.the tokens in it
        """
        assert isinstance(sent, str)

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
            tmp = ''
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
