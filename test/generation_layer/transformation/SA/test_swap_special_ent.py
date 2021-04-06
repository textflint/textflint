import unittest

from textflint.input_layer.component.sample.sa_sample import SASample
from textflint.generation_layer.transformation.SA.swap_special_ent \
    import SwapSpecialEnt

data = {'x': "Brilliant and moving performances by Tom "
             "Courtenay and Peter Finch",
        'y': 'negative'}
sa_sample = SASample(data)
swap_ins = SwapSpecialEnt(entity_type='person')


class TestSwapSpecialEnt(unittest.TestCase):
    def test_entity_type(self):
        self.assertEqual('movie', SwapSpecialEnt('movie').entity_type)
        self.assertEqual('person', SwapSpecialEnt('person').entity_type)
        self.assertRaises(ValueError, SwapSpecialEnt, 'person_movie')

    def test_get_entity_location(self):
        for tup_list in [[], "", {}]:
            self.assertEqual(len(swap_ins._get_entity_location(tup_list)), 0)

        # test data with multiple entity
        tup_list = sa_sample.concat_token(5)
        self.assertEqual(len(swap_ins._get_entity_location(tup_list)), 2)
        self.assertEqual(swap_ins._get_entity_location(tup_list),
                         [[5, 7], [8, 10]])

        # test get_entity_location with no entities
        test_data = {'x': "Brilliant and moving performances by and ",
                     'y': 'negative'}
        test_sample = SASample(test_data)
        tup_list = test_sample.concat_token(5)
        self.assertEqual([], swap_ins._get_entity_location(tup_list))

        # test data with entity at the beginning of sentence
        test_data = {'x': "Peter Finch Brilliant and moving "
                          "performances by and ",
                     'y': 'negative'}
        test_sample = SASample(test_data)
        tup_list = test_sample.concat_token(5)
        self.assertEqual(swap_ins._get_entity_location(tup_list), [[0, 2]])

        # test data with entity at the end of sentence
        test_data = {'x': " Brilliant and moving performances "
                          "by and Peter Finch",
                     'y': 'negative'}
        test_sample = SASample(test_data)
        tup_list = test_sample.concat_token(5)
        self.assertEqual(swap_ins._get_entity_location(tup_list), [[6, 8]])

    def test_SwapSpecialEnt(self):
        # test data with two type entities
        test_data = {'x': "Titanic Brilliant and moving performances "
                          "by Tom Courtenay and Peter Finch",
                     'y': 'negative'}
        test_sample = SASample(test_data)
        for entity_type in ['person', 'movie']:
            test_trans = SwapSpecialEnt(entity_type=entity_type)
            self.assertTrue(len(test_trans.transform(test_sample, n=5)) == 5)

        # test data with no entity
        test_data = {'x': "Brilliant and moving performances by  and ",
                     'y': 'negative'}
        test_sample = SASample(test_data)
        self.assertEqual([], test_trans.transform(test_sample))

        test_trans = SwapSpecialEnt(entity_type='person')
        # test data with entity at the beginning of sentence
        test_data = {'x': "Peter Finch Brilliant and moving "
                          "performances by and ",
                     'y': 'negative'}
        test_sample = SASample(test_data)
        trans_samples = test_trans.transform(test_sample)
        for mask_value in trans_samples[0].get_mask('x')[0:2]:
            self.assertEqual(mask_value, 2)

        # test data with entity at the end of sentence
        test_data = {'x': " Brilliant and moving performances "
                          "by and Peter Finch",
                     'y': 'negative'}
        test_sample = SASample(test_data)
        trans_samples = test_trans.transform(test_sample)
        for mask_value in \
                trans_samples[0].get_mask('x')[
                len(test_sample.get_words('x'))-2:]:
            self.assertEqual(mask_value, 2)


if __name__ == "__main__":
    unittest.main()
