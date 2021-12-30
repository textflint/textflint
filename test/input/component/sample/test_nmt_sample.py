import unittest

from textflint.input.component.sample.nmt_sample import *

data = {'source': "It's the most well-known Arab monument in the world..",
        'target': "Es ist das bekannteste arabische Bauwerk auf der ganzen Welt..."}
nmt_sample = NMTSample(data)


class TestNMTSample(unittest.TestCase): 
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, NMTSample, {'source': 'movie'})
        self.assertRaises(AssertionError, NMTSample, {'target': 'movie'})
        self.assertRaises(AssertionError, NMTSample, {'source': ''})
        self.assertRaises(AssertionError, NMTSample, {'source': 'movie', 'target': 5})
        self.assertRaises(AssertionError, NMTSample, {'source': [], 'target': []})
        self.assertRaises(AssertionError, NMTSample, {'source': 'the US', 'target': []})
        self.assertRaises(ValueError, NMTSample, {'source': 'the US', 'target': ''})
        self.assertRaises(ValueError, NMTSample, {'source': 'the US', 'target': 'the US'})
        self.assertRaises(ValueError, NMTSample, {'source': '', 'target': ''})

    def test_get_words(self):
        # test get words
        self.assertEqual(['It', "'s", 'the', 'most', 'well', '-', 'known', 'Arab', 'monument', 'in', 'the', 'world', '..'],
        nmt_sample.get_words('source'))
        self.assertEqual(['Es', 'ist', 'das', 'bekannteste', 'arabische', 'Bauwerk', 'auf', 'der', 'ganzen', 'Welt', '...'],
        nmt_sample.get_words('target'))

    def test_get_text(self):
        # test get text
        self.assertEqual("It's the most well-known Arab monument in the world..", nmt_sample.get_text('source'))
        self.assertEqual("Es ist das bekannteste arabische Bauwerk auf der ganzen Welt...", nmt_sample.get_text('target'))

    def test_get_value(self):
        # test get value
        self.assertEqual("It's the most well-known Arab monument in the world..", nmt_sample.get_value('source'))
        self.assertEqual("Es ist das bekannteste arabische Bauwerk auf der ganzen Welt...", nmt_sample.get_value('target'))

    def test_dump(self):
        # test dump
        self.assertEqual({'source': "It's the most well-known Arab monument in the world..",
                          'target': "Es ist das bekannteste arabische Bauwerk auf der ganzen Welt...",
                          'sample_id': None}, nmt_sample.dump())


if __name__ == "__main__":
    unittest.main()
