import unittest
import sys
import os
sys.path.insert(0, os.getcwd())
print(os.getcwd())
from textflint.input.component.sample.smcn_sample import *

data = {'sentence1': '我喜欢这本书。',
        'sentence2': '这本书是我讨厌的。',
        'y': '0'}
csm_sample = SMCNSample(data)


class TestSMSample(unittest.TestCase):
    def test_load_sample(self):
        # test wrong data
        self.assertRaises(AssertionError, SMCNSample,
                          {'sentence1': '我喜欢这本书。'})
        self.assertRaises(AssertionError, SMCNSample,
                          {'sentence2': '这本书是我讨厌的。'})
        self.assertRaises(AssertionError, SMCNSample,
                          {'x': '我喜欢这本书。'})
        self.assertRaises(AssertionError, SMCNSample, {'y': 'contradiction'})
        self.assertRaises(AssertionError, SMCNSample, {'sentence1': 11,
                                                     'sentence2': 22, 'y': '1'})
        self.assertRaises(AssertionError, SMCNSample, {'sentence1': [],
                                                     'sentence2': [], 'y': '0'})
        self.assertRaises(AssertionError, SMCNSample, {'sentence1': '',
                                                     'sentence2': ''})
        self.assertRaises(ValueError, SMCNSample, {
            'sentence1': '我喜欢这本书。',
            'sentence2': '这本书是我讨厌的。', 'y': 1})
    def test_dump(self):
        # test dump
        self.assertEqual({'sentence1': '我喜欢这本书。',
                          'sentence2': '这本书是我讨厌的。',
                          'y': '0',
                          'sample_id': None}, csm_sample.dump())



if __name__ == "__main__":
    unittest.main()
