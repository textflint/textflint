import unittest
from textflint.generation_layer.transformation.ABSA.add_diff import AddDiff
from textflint.input_layer.component.sample.absa_sample import ABSASample

data = {"sentence": "The food is great, I love their dumplings, cold sesame "
                    "noodles, chicken and shrimp dishs.",
        "term_list": {"32867472#663531#1_1": {"id": "32867472#663531#1_1",
                                              "polarity": "positive",
                                              "term": "food", "from": 4, "to": 8,
                                              "opinion_words": ["great"],
                                              "opinion_position": [[12, 17]]},
                      "32867472#663531#1_3": {"id": "32867472#663531#1_3",
                                              "polarity": "positive",
                                              "term": "dumplings", "from": 32,
                                              "to": 41, "opinion_words": ["love"],
                                              "opinion_position": [[21, 25]]},
                      "32867472#663531#1_2": {"id": "32867472#663531#1_2",
                                              "polarity": "positive",
                                              "term": "cold sesame noodles",
                                              "from": 43, "to": 62,
                                              "opinion_words": ["love"],
                                              "opinion_position": [[21, 25]]},
                      "32867472#663531#1_0": {"id": "32867472#663531#1_0",
                                              "polarity": "positive",
                                              "term": "chicken", "from": 64,
                                              "to": 71, "opinion_words": ["love"],
                                              "opinion_position": [[21, 25]]},
                      "32867472#663531#1_4": {"id": "32867472#663531#1_4",
                                              "polarity": "positive",
                                              "term": "shrimp dishs",
                                              "from": 76, "to": 88,
                                              "opinion_words": ["love"],
                                              "opinion_position": [[21, 25]]}}}
absa_sample = ABSASample(data)
AddDiff = AddDiff()


class TestAddDiff(unittest.TestCase):

    def test_transform(self):
        extra_text = {
            'positive': [('apple', ['apple is good']),
                         ('orange', ['orange is good']),
                         ('banana', ['banana is good'])],
            'negative': [('apple', ['apple is not good']),
                         ('orange', ['orange is not good']),
                         ('banana', ['banana is not good'])],
            'neutral': [('apple', ['apple is red']),
                        ('orange', ['orange is orange']),
                        ('banana', ['banana is yellow'])]}
        trans = AddDiff.transform(absa_sample, n=1,
                                  field='sentence', extra_text=extra_text)

        term_id = [idx for idx in data['term_list']]
        for i, trans_sample in enumerate(trans):
            trans_data = trans_sample.dump()
            ori_sentence = data['sentence']
            trans_sentence = trans_data['sentence']
            ori_polarity = data['term_list'][term_id[i]]['polarity']
            ori_opinion_words = data['term_list'][term_id[i]]['opinion_words']
            trans_id = trans_data['trans_id']
            trans_polarity = trans_data['term_list'][trans_id]['polarity']
            trans_opinion_words = trans_data['term_list'][trans_id]['opinion_words']
            self.assertTrue(ori_sentence != trans_sentence)
            self.assertTrue(ori_polarity == trans_polarity)
            self.assertTrue(ori_opinion_words == trans_opinion_words)


if __name__ == "__main__":
    unittest.main()
