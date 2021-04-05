import unittest

from TextFlint.input_layer.component.field.cn_text_field import *


class TestCnTextField(unittest.TestCase):
    def test_cn_text_field(self):
        self.assertRaises(ValueError, CnTextField, {})

        # test mask
        test_field = CnTextField('小明去上海')

        # test pos tag only the return format is correct, not the label
        pos_tag = test_field.pos_tags()
        self.assertEqual(pos_tag[-1][-1] + 1, len(test_field.token))
        self.assertEqual(pos_tag[0][1], 0)
        for tag in pos_tag:
            self.assertTrue([str, int, int], [type(i) for i in tag])

        # test ner only the return format is correct, not the label
        ner, ner_label = test_field.ner()
        self.assertEqual(len(ner_label), len(test_field.token))
        for tag in ner:
            self.assertTrue([str, int, int] == [type(k) for k in tag] and tag[1] <= tag[2])


if __name__ == "__main__":
    unittest.main()
