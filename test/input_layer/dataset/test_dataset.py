import unittest

from TextFlint.input_layer.dataset import *
from test.settings import *


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        data1 = Dataset(task='SA')
        data2 = Dataset(task='SA')
        data3 = Dataset(task='SA')
        data4 = Dataset(task='SA')

        # test load csv
        data1.load_csv(CSV_DATA_PATH, headers=['y', 'x'])
        self.assertEqual(101, len(data1))

        # test load json
        data2.load_json(JSON_DATA_PATH)
        self.assertEqual(101, len(data2))
        for i, j in zip(data1, data2):
            self.assertEqual(i.get_value('x'), j.get_value('x'))
            self.assertEqual(i.get_value('y'), j.get_value('y'))

        # test save csv and json
        data2.save_csv(TEST_CSV_DATA_PATH)
        data1.save_json(TEST_JSON_DATA_PATH)
        data3.load_csv(TEST_CSV_DATA_PATH)
        data4.load_json(TEST_JSON_DATA_PATH)
        for i, j, k in zip(data1, data3, data4):
            self.assertEqual(i.get_value('x'), j.get_value('x'))
            self.assertEqual(i.get_value('y'), j.get_value('y'))
            self.assertEqual(i.get_value('x'), k.get_value('x'))
            self.assertEqual(i.get_value('y'), k.get_value('y'))
        os.remove(TEST_JSON_DATA_PATH)
        os.remove(TEST_CSV_DATA_PATH)


if __name__ == '__main__':
    unittest.main()
