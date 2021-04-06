import unittest
from textflint.generation_layer.transformation.ABSA.rev_tgt import RevTgt
from textflint.input_layer.component.sample.absa_sample import ABSASample

data = {"sentence": "The bread is top notch as well.",
       "term_list": {
           "32897564#894393#2_0": 
           {"id": "32897564#894393#2_0", 
           "polarity": "positive", 
           "term": "bread", 
           "from": 4, 
           "to": 9, 
           "opinion_words": ["top notch"], 
           "opinion_position": [[13, 22]]}
           }
        }   

absa_sample = ABSASample(data)
reverse_target = RevTgt()


class TestRevTgt(unittest.TestCase):
    
    def test_transform(self):
        trans = reverse_target.transform(absa_sample, n=1, field='sentence')
        term_id = [idx for idx in data['term_list']]
        for i, trans_sample in enumerate(trans):
            trans_data = trans_sample.dump()
            ori_polarity = data['term_list'][term_id[i]]['polarity']
            ori_opinion_words = data['term_list'][term_id[i]]['opinion_words']
            trans_id = trans_data['trans_id']
            trans_polarity = trans_data['term_list'][trans_id]['polarity']
            trans_opinion_words = trans_data['term_list'][trans_id]['opinion_words']
            self.assertTrue(ori_polarity != trans_polarity)
            self.assertTrue(ori_opinion_words != trans_opinion_words)


if __name__ == "__main__":
    unittest.main()
