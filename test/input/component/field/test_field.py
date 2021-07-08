import unittest

from textflint.input.component.field.field import *


class TestField(unittest.TestCase):
    def test_field(self):
        test_field = Field(10, type(10))

        self.assertEqual(10, test_field.field_value)
        self.assertEqual(type(10), test_field.field_type)
        test_field.field_value = 15
        self.assertEqual(15, test_field.field_value)


if __name__ == "__main__":
    unittest.main()
