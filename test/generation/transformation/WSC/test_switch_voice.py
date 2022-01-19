import unittest

from textflint.input.component.sample.wsc_sample import WSCSample
from textflint.generation.transformation.WSC import SwitchVoice

sample1 = WSCSample({"text": "The city councilmen refused the demonstrators a permit because they feared violence.",
           "target": {"noun1": "The city councilmen", "noun2": "The demonstrators",
                      "noun1_idx": 0, "noun2_idx": 4, "pron": "they", "pron_idx": 9},
           "label": 0, "index": 0})
wsc_antonymswap = SwitchVoice()


class TestNliSwapAnt(unittest.TestCase):
    def test_whether_changed(self):
        # test whether the sample changed or not
        trans = wsc_antonymswap.transform(sample1)
        if trans != []:
            self.assertTrue(sample1.text.text !=
                            trans[0].text.text)



if __name__ == "__main__":
    unittest.main()
