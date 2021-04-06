import unittest
from textflint.generation_layer.transformation.ABSA.rev_non import RevNon
from textflint.input_layer.component.sample.absa_sample import ABSASample

data = {"sentence": "The food is great, I love their dumplings, "
                    "cold sesame noodles, chicken and shrimp dishs.",
        "term_list": {
            "32867472#663531#1_1": {"id": "32867472#663531#1_1",
                                    "polarity": "positive",
                                    "term": "food", "from": 4,
                                    "to": 8, "opinion_words": ["great"],
                                    "opinion_position": [[12, 17]]},
            "32867472#663531#1_3": {"id": "32867472#663531#1_3",
                                    "polarity": "positive", "term": "dumplings",
                                    "from": 32, "to": 41, "opinion_words":
                                        ["love"], "opinion_position": [[21, 25]]},
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
reverse_target = RevNon()


class TestRevNon(unittest.TestCase):

    def test_transform(self):
        trans = reverse_target.transform(absa_sample, n=1, field='sentence')
        all_term_id = [idx for idx in data['term_list']]

        for i, trans_sample in enumerate(trans):
            trans_data = trans_sample.dump()
            trans_id = trans_data['trans_id']
            other_ids = all_term_id
            other_ids.remove(trans_id)

            for other_id in other_ids:
                ori_polarity = data['term_list'][other_id]['polarity']
                ori_opinion_words = data['term_list'][other_id]['opinion_words']
                trans_polarity = trans_data['term_list'][other_id]['polarity']
                trans_opinion_words = trans_data['term_list'][other_id]['opinion_words']
                self.assertTrue(ori_polarity != trans_polarity)
                self.assertTrue(ori_opinion_words != trans_opinion_words)


if __name__ == "__main__":
    unittest.main()
