import unittest

from textflint.input_layer.component.sample.sa_sample import SASample
from textflint.generation_layer.transformation.SA.add_sum import AddSum

data = {'x': "Brilliant and moving performances by Tom Courtenay "
             "and Peter Finch",
        'y': 'positive'}
sa_sample = SASample(data)
swap_ins = AddSum(entity_type='person')


class TestAddSum(unittest.TestCase):
    def test_entity_type(self):
        self.assertEqual('movie', AddSum('movie').entity_type)
        self.assertEqual('person', AddSum('person').entity_type)
        self.assertRaises(ValueError, AddSum, 'person_movie')

    def test_get_insert_info(self):
        for tup_list in [[], "", {}]:
            self.assertEqual(len(swap_ins._get_insert_info(tup_list)[0]), 0)
        tup_list = sa_sample.concat_token(5)
        self.assertEqual(len(swap_ins._get_insert_info(tup_list)[0]),
                         len(swap_ins._get_insert_info(tup_list)[1]))

    def test_AddSum(self):
        # test data with two type entities
        test_data = {'x': "Titanic Brilliant and moving performances "
                          "by Tom Courtenay and Peter Finch",
                     'y': 'negative'}
        test_sample = SASample(test_data)
        for entity_type in ['person', 'movie']:
            test_trans = AddSum(entity_type=entity_type)
            self.assertEqual(len(test_trans.transform(test_sample)), 1)

        # test data with no entity
        test_data = {'x': "Brilliant and moving performances by  and ",
                     'y': 'positive'}
        test_sample = SASample(test_data)
        self.assertEqual([], test_trans.transform(test_sample))

        # test data with entity at the end of sentence
        test_data = {'x': "Brilliant and moving performances by Peter Finch",
                     'y': 'positive'}
        test_sample = SASample(test_data)
        test_trans = AddSum(entity_type='person')
        trans_samples = test_trans.transform(test_sample)
        self.assertTrue(len(trans_samples[0].get_words('x')) >
                        len(test_sample.get_words('x')))
        for mask_value in \
                trans_samples[0].get_mask('x')[len(test_sample.get_words('x')):]:
            self.assertEqual(mask_value, 2)

        # test data with entity at the beginning of sentence
        test_data = {'x': "Peter Finch Brilliant and moving performances by",
                     'y': 'positive'}
        test_sample = SASample(test_data)
        test_trans = AddSum(entity_type='person')
        trans_samples = test_trans.transform(test_sample)
        self.assertTrue(len(trans_samples[0].get_words('x')) >
                        len(test_sample.get_words('x')))
        for mask_value in trans_samples[0].get_mask('x')[2:3]:
            self.assertEqual(mask_value, 2)


if __name__ == "__main__":
    unittest.main()
