from textflint.common.utils.fp_utils import concat
from textflint.input_layer.component.sample.coref_sample import *

class CorefDebug:

    @staticmethod
    def words(s):
        """
        Usage: word("i love u .") == ["i", "love", "u", "."]
        :param str s: the sentence to be splitted
        :return list: the splitted sentence as a word list
        """
        import re
        s = s.strip()
        if s == "":
            return []
        s = re.sub('\s+', ' ', s)
        s = re.sub('\t', ' ', s)
        s = re.sub('\n', ' ', s)
        return s.split(" ")

    @staticmethod
    def unwords(ws):
        """
        Usage: unwords(["i", "love", "u", "."]) == "i love u ."
        :param list ws: the word list to be joined
        :return str: the joined sentence
        """
        return " ".join(ws)

    @staticmethod
    def make_sample_from_sentences_and_clusters(
            sens, clusters, identifier="", conllstyle=False):
        r""" Make a sample (in the form of conll-style dict) with `sentences` and 
            `clusters` given. This method is useful for making debugging samples.

        :param list sens: sentences. word list list
        :param str sign: str, will be added to label
        :param str identifier: str, optional
                make `speakers` and `doc_key` different among different docs, 
                to identify different samples.
        :param bool conllstyle: 
                if True, returns a conll-style dict
                if False, returns a CorefSample
        :return dict: A conll-style dict
        """

        def make_cons(sens, sign):
            r""" By given sentences, make sound `constituents` and `ner`
                corresponding to the given sentence. 
                This method is used in `make_sample_from_sentences_and_clusters`
                for making a conll-style dict from sentences. 
            :param list sens: sentences. word list list
            :param str sign: str, will be added to label
            :return list: sound constituents/ner with the same form as conll["ner"]
            """
            x = concat(sens)
            return [[i, i, str(i)+"-"+x[i]+"-"+sign] for i in range(len(x))]

        ret_conll = {
            "speakers": [["sp"+identifier for w in sen] for sen in sens],
            "doc_key": "doc_key" + identifier,
            "sentences": sens,
            "constituents": make_cons(sens, "C"),
            "clusters": clusters,
            "ner": make_cons(sens, "N")
        }

        if conllstyle:
            return ret_conll
        else:
            return CorefSample(ret_conll)

    @staticmethod
    def coref_sample1():
        r""" Returns a debug sample for coref.
        :return ~textflint.input_layer.component.sample.CorefSample: A CorefSample
        """
        sens = [
            CorefDebug.words("I love my pet Anna ."),
            CorefDebug.words("She is my favorite .")
        ]
        clusters = [[[2, 3], [4, 4], [6, 6]]]
        return CorefDebug.make_sample_from_sentences_and_clusters(
            sens, clusters, "1")

    @staticmethod
    def coref_sample2():
        r""" Returns a debug sample for coref.
        :return ~textflint.input_layer.component.sample.CorefSample: A CorefSample
        """
        sens = [
            CorefDebug.words("Bob 's wife Anna likes winter ."),
            CorefDebug.words("However , he loves summer .")
        ]
        clusters = [[[0, 2], [3, 3]], [[0, 0], [9, 9]]]
        return CorefDebug.make_sample_from_sentences_and_clusters(
            sens, clusters, "2")

    @staticmethod
    def coref_sample3():
        r""" Returns a debug sample for coref.
        :return ~textflint.input_layer.component.sample.CorefSample: A CorefSample
        """
        sens = [CorefDebug.words("Nothing .")]
        clusters = []
        return CorefDebug.make_sample_from_sentences_and_clusters(
            sens, clusters, "3")

    @staticmethod
    def coref_sample4():
        r""" Returns a debug sample for coref.
        :return ~textflint.input_layer.component.sample.CorefSample: A CorefSample
        """
        sens = [
            CorefDebug.words(
                "The quick brown fox jumps over eleven lazy dogs in one hour 50 minutes ."),
            CorefDebug.words("The fox is better than the dogs .")
        ]
        clusters = [[[0, 3], [15, 16]], [[6, 8], [20, 21]]]
        return CorefDebug.make_sample_from_sentences_and_clusters(
            sens, clusters, "4")

    @staticmethod
    def coref_sample5():
        r""" Returns a debug sample for coref.
        :return ~textflint.input_layer.component.sample.CorefSample: A CorefSample
        """
        return CorefSample(None)

    @staticmethod
    def coref_sample6():
        r""" Returns a debug sample for coref.
        :return ~textflint.input_layer.component.sample.CorefSample: A CorefSample
        """
        sens = [
            CorefDebug.words("textflint ! @ # $ % ^ & * J ) 1321"),
            CorefDebug.words("Jotion lives in Flint 105 kilometers away .")
        ]
        clusters = [[[0, 0], [15, 15]], [[9, 9], [12, 12]]]
        return CorefDebug.make_sample_from_sentences_and_clusters(
            sens, clusters, "6")
