#!/usr/bin/python3
# -*- coding: utf-8 -*-
import pickle

from ..settings import *
from ..utils.install import download_if_needed


class ModelManager(object):
    def __init__(self):
        self.loaded_models = {}
        self.model_func_mapping = {
            DP_PARSER: self.get_dp_parser,
            CFG_PARSER: self.get_cfg_parser,
            NLTK_WORDNET: self.get_nltk_wordnet,
            NLTK_WORDNET_DELEMMA: self.get_nltk_wordnet_delemma,
            STANFORD_NER: self.get_stanford_ner,
            STANFORD_POS_TAGGER: self.get_stanford_tagger
        }

    def load(self, name):
        assert name in self.model_func_mapping, "Unsupported model name!"
        if name in self.loaded_models:
            return self.loaded_models[name]
        path = download_if_needed(name)
        model = self.model_func_mapping[name](path)
        self.loaded_models[name] = model
        return model

    @staticmethod
    def get_dp_parser(path):
        return (
            __import__("nltk.parse.stanford")
            .parse.stanford.StanfordDependencyParser(
                path_to_jar=os.path.join(
                    path,
                    './stanford-parser.jar'
                ),
                path_to_models_jar=os.path.join(
                    path,
                    "./stanford-parser-4.2.0-models.jar"
                ),
                model_path=os.path.join(
                    path,
                    './englishPCFG.ser.gz'
                )
            ).raw_parse)

    @staticmethod
    def get_cfg_parser(path):
        return (
            __import__("nltk.parse.stanford")
            .parse.stanford.StanfordParser(
                path_to_jar=os.path.join(
                    path,
                    './stanford-parser.jar'
                ),
                path_to_models_jar=os.path.join(
                    path,
                    "./stanford-parser-4.2.0-models.jar"
                ),
                model_path=os.path.join(
                    path,
                    './englishPCFG.ser.gz'
                )

            ).raw_parse)

    @staticmethod
    def get_nltk_wordnet(path):
        wnc = __import__("nltk").corpus.WordNetCorpusReader(path, None)

        def lemma(word, pos):
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
            if pp is None:  # do not need lemmatization
                return word
            lemmas = wnc._morphy(word, pp)
            return min(lemmas, key=len) if len(lemmas) > 0 else word

        def all_lemma(pos):
            return wnc.all_lemma_names(pos)

        wnc.lemma = lemma
        wnc.all_lemma = all_lemma
        return wnc

    @staticmethod
    def get_nltk_wordnet_delemma(path):
        return pickle.load(open(path, "rb"))

    @staticmethod
    def get_stanford_ner(path):
        return (
            __import__("nltk").StanfordNERTagger(
                model_filename=os.path.join(
                    path,
                    "english.muc.7class.distsim.crf.ser.gz"
                ),
                path_to_jar=os.path.join(
                    path,
                    "stanford-ner.jar"
                ),
            ).tag
        )

    @staticmethod
    def get_stanford_tagger(path):
        return (__import__("nltk").tag.StanfordPOSTagger(
            model_filename=os.path.join(
                path,
                'english-bidirectional-distsim.tagger'
            ),
            path_to_jar=os.path.join(
                path,
                'stanford-postagger-4.2.0.jar')
            )
        ).tag
