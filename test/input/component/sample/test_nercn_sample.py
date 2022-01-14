import unittest

from textflint.input.component.sample.nercn_sample import *

data = {'x':'上海浦东开发与法制建设同步', 'y':['B-GPE','E-GPE','B-GPE','E-GPE','O','O','O','O','O','O','O','O','O']}
cnner_sample = NERCnSample(data)


class TestNERCnSample(unittest.TestCase):
    

    def test_load(self):
        self.assertRaises(AssertionError, NERCnSample, {'x': '上海浦东开发与法制建设同步'})
        self.assertRaises(AssertionError, NERCnSample, {'y': '上海浦东开发与法制建设同步'})
        self.assertRaises(AssertionError, NERCnSample, {'x': ''})
        self.assertRaises(AssertionError, NERCnSample,
                          {'x': ['上','海','浦','东','开','发','与','法','制','建','设','同','步'],
                           'y': ['E-GPE','E-GPE','B-GPE','E-GPE','O','O','O','O','O','O','O','O','O']})
        self.assertRaises(AssertionError, NERCnSample,{'x': ['上','海','浦','东','开','发','与','法','制','建','设','同','步'],
                           'y': ['B-GPE','E-GPE','B-GPE','E-GPE','O','O','O','O','O','O','O','O']})



    def test_dump(self):
        self.assertEqual({'x': ['上', '海', '浦', '东', '开', '发', '与', '法', '制', '建', '设', '同', '步'],
                            'y': ['B-GPE','E-GPE','B-GPE','E-GPE','O','O','O','O','O','O','O','O','O'],
                           'sample_id': None},
                         cnner_sample.dump())



    def test_find_entities_BMOES(self):
        entities = cnner_sample.find_entities_BMOES(cnner_sample.dump()['x'],
                                                cnner_sample.dump()['y'])
        self.assertEqual([{'start': 0, 'end': 1, 'entity': '上海', 'tag': 'GPE'},
                          {'start': 2, 'end': 3, 'entity': '浦东', 'tag': 'GPE'}], entities)

    def test_entities_replace(self):
        self.assertRaises(AssertionError, cnner_sample.entities_replace,
                          [], ['啥也不是'])
        self.assertRaises(AssertionError, cnner_sample.entities_replace,
                          ['啥也不是'], [])
        self.assertRaises(AssertionError, cnner_sample.entities_replace,
                          ['啥也不是'], '')

        replaced = cnner_sample.entities_replace([{'start': 2, 'end': 3, 'entity': '浦东', 'tag': 'GPE'}],['浦东新区'])
        self.assertEqual({'x': ['上','海','浦','东','新','区','开','发','与','法','制','建','设','同','步'],
                          'y': ['B-GPE','E-GPE','B-GPE','M-GPE','M-GPE','E-GPE','O','O','O','O','O','O','O','O','O'],
                          'sample_id': None}, replaced.dump())    


if __name__ == "__main__":  
    unittest.main()