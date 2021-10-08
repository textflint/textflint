import unittest

from textflint.input.component.sample.wsd_sample import *
from textflint.common.preprocess.en_processor import EnProcessor


sentence = ["Your", "Oct.", "6", "editorial", "``", "The", "Ill",
            "Homeless", "``", "referred", "to",
            "research", "by", "us", "and", "six", "of", "our",
            "colleagues", "that", "was", "reported", "in",
            "the", "Sept.", "8", "issue", "of", "the", "Journal", "of",
            "the", "American", "Medical",
            "Association", "."]
pos = ["PRON", "NOUN", "NUM", "NOUN", ".", "DET", "NOUN", "NOUN", ".", "VERB",
       "X", "NOUN", "ADP", "PRON", "CONJ", "NUM", "ADP", "PRON", "NOUN", "ADP",
       "VERB", "VERB", "ADP", "DET", "NOUN", "NUM", "NOUN", "ADP", "DET",
       "NOUN", "ADP", "DET", "ADJ", "ADJ", "NOUN", "."]
lemma = ["Your", "Oct.", "6", "editorial", "``", "The", "Ill",
         "Homeless", "``", "refer", "to", "research", "by", "us",
         "and", "six", "of", "our", "colleague", "that", "be",
         "report", "in", "the", "Sept.", "8", "issue", "of", "the",
         "Journal", "of", "the", "American", "Medical",
         "Association", "."]
instance = [["d000.s000.t000", 9, 10, "referred", "refer%2:32:01::"],
            ["d000.s000.t001", 11, 12, "research",
             "research%1:04:00::"],
            ["d000.s000.t002", 21, 22, "reported",
             "report%2:32:04::"]]
data = {
    "sentence": sentence,
    "pos": pos,
    "lemma": lemma,
    "instance": instance,
    "sentence_id": "d000.s000",
    "source": "semeval2007", "sample_id": None}


class TestWSDSample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data

        # invalid field(assertion error)
        self.assertRaises(
            AssertionError, WSDSample,
            {
                "x": sentence,
                "pos": pos,
                "lemma": lemma,
                "instance": instance,
                "sentence_id": "d000.s000",
                "source": "semeval2007", "sample_id": None})

        # miss field(assertion error)
        self.assertRaises(
            AssertionError, WSDSample,
            {
                "sentence": sentence,
                "pos": pos,
                "lemma": lemma,
                "instance": instance,
                "source": "semeval2007", "sample_id": None})
        # wrong type
        self.assertRaises(
            AssertionError, WSDSample,
            {
                "sentence": EnProcessor.inverse_tokenize(sentence),
                "pos": pos,
                "lemma": lemma,
                "instance": instance,
                "sentence_id": "d000.s000",
                "source": "semeval2007", "sample_id": None})

        self.assertRaises(
            AssertionError, WSDSample,
            {
                "sentence": sentence,
                "pos": pos,
                "lemma": lemma,
                "instance": [[t[0], t[1] - 1, t[2], t[3], t[4]] for t in
                             instance],
                "sentence_id": "d000.s000",
                "source": "semeval2007", "sample_id": None})

        self.assertRaises(
            AssertionError, WSDSample,
            {
                "sentence": sentence,
                "pos": pos,
                "lemma": lemma,
                "instance": [[t[0], t[1], t[2] + 1, t[3], t[4]] for t in
                             instance],
                "sentence_id": "d000.s000",
                "source": "semeval2007", "sample_id": None})

        self.assertRaises(
            AssertionError, WSDSample,
            {
                "sentence": sentence,
                "pos": pos,
                "lemma": lemma,
                "instance": [[t[0], t[2], t[1], t[3], t[4]] for t in
                             instance],
                "sentence_id": "d000.s000",
                "source": "semeval2007", "sample_id": None})

    def test_tag_words(self):
        wsd_sample = WSDSample(data)
        tag_list = wsd_sample.tag_words(sentence, wn=False)
        self.assertEqual(tag_list, pos)

    def test_get_lemma(self):
        wsd_sample = WSDSample(data)
        lemma_list = wsd_sample.get_lemma(sentence)
        self.assertEqual(lemma_list, lemma)

    def test_insert_field_before_indices(self):
        wsd_sample = WSDSample(data)
        # test insert before index and mask
        ins_sample = wsd_sample.insert_field_before_indices("sentence",
                                                            [9, 21], ["here",
                                                                      ["and",
                                                                       "then"],
                                                                      ])
        new_sent = ["Your", "Oct.", "6", "editorial", "``", "The", "Ill",
                    "Homeless", "``", "here", "referred", "to",
                    "research", "by", "us", "and", "six", "of", "our",
                    "colleagues", "that", "was", "and", "then", "reported",
                    "in",
                    "the", "Sept.", "8", "issue", "of", "the", "Journal", "of",
                    "the", "American", "Medical",
                    "Association", "."]
        # check sentence
        self.assertEqual(EnProcessor.inverse_tokenize(new_sent),
                         ins_sample.get_text("sentence"))
        # check pos
        tag_list = wsd_sample.tag_words(new_sent, wn=False)
        self.assertEqual(tag_list,
                         ins_sample.get_value("pos"))
        # check lemma
        lemma_list = wsd_sample.get_lemma(new_sent)
        self.assertEqual(lemma_list, ins_sample.get_value("lemma"))

    def test_insert_field_after_indices(self):
        wsd_sample = WSDSample(data)
        # test insert after index and mask
        ins_sample = wsd_sample.insert_field_after_indices("sentence",
                                                           [9, 21], ["here",
                                                                     ["and",
                                                                      "then"],
                                                                     ])
        # check sentence
        new_sent = ["Your", "Oct.", "6", "editorial", "``", "The", "Ill",
                    "Homeless", "``", "referred", "here", "to",
                    "research", "by", "us", "and", "six", "of", "our",
                    "colleagues", "that", "was", "reported", "and", "then",
                    "in",
                    "the", "Sept.", "8", "issue", "of", "the", "Journal", "of",
                    "the", "American", "Medical",
                    "Association", "."]
        self.assertEqual(EnProcessor.inverse_tokenize(new_sent),
                         ins_sample.get_text("sentence"))
        # check pos
        tag_list = wsd_sample.tag_words(new_sent, wn=False)
        self.assertEqual(tag_list,
                         ins_sample.get_value("pos"))
        # check lemma
        lemma_list = wsd_sample.get_lemma(new_sent)
        self.assertEqual(lemma_list, ins_sample.get_value("lemma"))

    def test_delete_field_at_indices(self):
        wsd_sample = WSDSample(data)
        # test delete and mask
        del_sample = wsd_sample.delete_field_at_indices("sentence",
                                                        [9, [11, 13], 21])
        # check sentece
        new_sent = ["Your", "Oct.", "6", "editorial", "``", "The",
                    "Ill",
                    "Homeless", "``", "to",
                    "us", "and", "six", "of", "our",
                    "colleagues", "that", "was", "in",
                    "the", "Sept.", "8", "issue", "of", "the",
                    "Journal", "of",
                    "the", "American", "Medical",
                    "Association", "."]
        self.assertEqual(
            new_sent,
            del_sample.get_value("sentence"))
        # check pos
        tag_list = wsd_sample.tag_words(new_sent, wn=False)
        self.assertEqual(tag_list,
                         del_sample.get_value("pos"))
        # check lemma
        lemma_list = wsd_sample.get_lemma(new_sent)
        self.assertEqual(lemma_list, del_sample.get_value("lemma"))

    def test_replace_field_at_indices(self):
        wsd_sample = WSDSample(data)
        rep_sample = wsd_sample.replace_field_at_indices("sentence",
                                                         [9, [11, 13], 21],
                                                         ["equal",
                                                          ["investigation",
                                                           "from"],
                                                          "here"])
        # check sentence
        new_sent = ["Your", "Oct.", "6", "editorial", "``", "The", "Ill",
                    "Homeless", "``", "equal", "to",
                    "investigation", "from", "us", "and", "six", "of", "our",
                    "colleagues", "that", "was", "here", "in",
                    "the", "Sept.", "8", "issue", "of", "the", "Journal", "of",
                    "the", "American", "Medical",
                    "Association", "."]
        self.assertEqual(EnProcessor.inverse_tokenize(new_sent),
                         rep_sample.get_text("sentence"))
        # check pos
        tag_list = wsd_sample.tag_words(new_sent, wn=False)
        self.assertEqual(tag_list,
                         rep_sample.get_value("pos"))
        # check lemma
        lemma_list = wsd_sample.get_lemma(new_sent)
        self.assertEqual(lemma_list, rep_sample.get_value("lemma"))

    def test_unequal_replace_field_at_indices(self):
        wsd_sample = WSDSample(data)
        # test unequal replace before index and mask
        rep_sample = wsd_sample.unequal_replace_field_at_indices("sentence",
                                                                 [[9, 11],
                                                                  [16, 17]],
                                                                 [[
                                                                     "refer%2:32:04::"],
                                                                     [
                                                                         "bring_up%2:32:04::"]
                                                                 ])
        # check sentence
        new_sent = ["Your", "Oct.", "6", "editorial", "``", "The", "Ill",
                    "Homeless", "``", "refer",
                    "research", "by", "us", "and", "six", "bring", "up", "our",
                    "colleagues", "that", "was", "reported", "in",
                    "the", "Sept.", "8", "issue", "of", "the", "Journal", "of",
                    "the", "American", "Medical",
                    "Association", "."]
        self.assertEqual(EnProcessor.inverse_tokenize(new_sent),
                         rep_sample.get_text("sentence"))
        # check pos
        gold_list = ["PRON", "NOUN", "NUM", "NOUN", ".", "DET", "NOUN", "NOUN",
                     ".", "VERB", "NOUN", "ADP", "PRON", "CONJ", "NUM", "ADP",
                     "ADP", "PRON", "NOUN", "ADP", "VERB", "VERB", "ADP", "DET",
                     "NOUN", "NUM", "NOUN", "ADP", "DET", "NOUN", "ADP", "DET",
                     "ADJ", "ADJ", "NOUN", "."]
        self.assertEqual(rep_sample.get_value("pos"), gold_list)
        # check lemma
        gold_list = ["Your", "Oct.", "6", "editorial", "``", "The", "Ill",
                     "Homeless", "``", "refer", "research", "by", "us", "and",
                     "six", "bring_up", "bring_up", "our", "colleague", "that",
                     "be", "report", "in", "the", "Sept.", "8", "issue", "of",
                     "the", "Journal", "of", "the", "American", "Medical",
                     "Association", "."]
        self.assertEqual(gold_list, rep_sample.get_value("lemma"))

    def test_mask(self):
        wsd_sample = WSDSample(data)
        # test does the mask label work
        new = wsd_sample.insert_field_after_index("sentence", 0, "Emmm")
        new = new.delete_field_at_index("sentence", 1)
        # TODO wait repair bug

    def test_get_words(self):
        wsd_sample = WSDSample(data)
        # test get words
        self.assertEqual(sentence, wsd_sample.get_words("sentence"))

    def test_get_text(self):
        wsd_sample = WSDSample(data)
        # test get text
        self.assertEqual(EnProcessor.inverse_tokenize(sentence),
                         wsd_sample.get_text("sentence"))

    def test_get_value(self):
        wsd_sample = WSDSample(data)
        # test get value
        self.assertEqual(sentence,
                         wsd_sample.get_value("sentence"))

    def test_dump(self):
        # test dump
        wsd_sample = WSDSample(data)
        self.assertEqual(data,
                         wsd_sample.dump())


if __name__ == "__main__":
    unittest.main()
