"""Unittests for ReportGenerator."""
import unittest

import pandas as pd

from textflint.report_layer.analyzer.analyzer import *
from textflint.report_layer.report_generator.report_generator \
    import BarChart, ReportGenerator

test_evaluate_result = {
    "model_name": "BERT-base(cased)",
    "dataset_name": "CONLL2003",
    "transformation": {
        "Case": {
            "ori_f1": 0.9115,
            "trans_f1": 0.4082,
            "size": 3446,
        },
        "Keyboard": {
            "ori_f1": 0.9115,
            "trans_f1": 0.79,
            "size": 3443,
        },
        "MLMSuggestion": {
            "ori_f1": 0.9212,
            "trans_f1": 0.8695,
            "size": 3204,
        },
        "Number": {
            "ori_f1": 0.9382,
            "trans_f1": 0.9376,
            "size": 1124,
        },
        "Ocr": {
            "ori_f1": 0.9117,
            "trans_f1": 0.8135,
            "size": 3432,
        },
        "ReverseNeg": {
            "ori_f1": 0.9125,
            "trans_f1": 0.8952,
            "size": 2177,
        },
        "SpellingError": {
            "ori_f1": 0.9193,
            "trans_f1": 0.7644,
            "size": 2681,
        },
        "Tense": {
            "ori_f1": 0.9193,
            "trans_f1": 0.9189,
            "size": 1265,
        },
        "SwapAnt": {
            "ori_f1": 0.9129,
            "trans_f1": 0.8883,
            "size": 834,
        },
        "SwapSyn": {
            "ori_f1": 0.9119,
            "trans_f1": 0.8538,
            "size": 834,
        },
        "Twitter": {
            "ori_f1": 0.9115,
            "trans_f1": 0.8942,
            "size": 3446,
        },
        "Typos": {
            "ori_f1": 0.9115,
            "trans_f1": 0.7881,
            "size": 3433,
        },
        "InsertAdv": {
            "ori_f1": 0.9175,
            "trans_f1": 0.9063,
            "size": 1475,
        },

        "AppendIrr": {
            "ori_f1": 0.9115,
            "trans_f1": 0.8701,
            "size": 3446,
        },
        "SwapContraction": {
            "ori_f1": 0.9263,
            "trans_f1": 0.925,
            "size": 190,
        },
        "Punctuation": {
            "ori_f1": 0.9115,
            "trans_f1": 0.886,
            "size": 3443,
        },

    },
    "subpopulation": {
        "LengthLengthSubPopulation-0.0-0.2": {
            "trans_f1": 0.8984,
            "size": 688
        },
        "PrejudiceSubPopulation-woman": {
            "trans_f1": 0.8956,
            "size": 688,
        }
    },
    "attack": {
        "Bert-Attack": {
            "ori_f1": 0.9142,
            "trans_f1": 0.24,
            "size": 2311,
        }
    }
}

evaluate_result = {
    "transformation": {
        "Case": {
            "size": 25000,
            "ori_accuracy": 0.9617,
            "trans_accuracy": 0.8596,
        },
        "InsertAdv": {
            "ori_accuracy": 0.9618,
            "trans_accuracy": 0.957,
            "size": 24996
        },
        "SwapSyn": {
            "ori_accuracy": 0.9617,
            "trans_accuracy": 0.9592,
            "size": 25000
        },
        "MLMSuggestion": {
            "ori_accuracy": 0.9617,
            "trans_accuracy": 0.9554,
            "size": 24049
        },
        "BackTrans": {
            "ori_accuracy": 0.9617,
            "trans_accuracy": 0.636,
            "size": 25000
        }
    },
    "subpopulation": {
        "LengthSubPopulation-0%-20%": {
            "trans_accuracy": 0.9578,
            "size": 5144
        },
        "LengthSubPopulation-80%-100%": {
            "trans_accuracy": 0.3786127167630058,
            "size": 5144
        }
    },
    "attack": {
        "Bert-Attack": {
            "size": 500,
            "ori_accuracy": 0.9617,
            "trans_accuracy": 0.5413,
        }
    },
    "model_name": "XLNET",
    "dataset_name": "IMDB"
}


# class TestBarChart(unittest.TestCase):
#     def setUp(self):
#         self.cols = [
#             ScoreColumn("f1", 0, 1, is_0_to_1=True),
#             ScoreColumn("perplexity", 0, 50),
#             NumericColumn("Size"),
#         ]
#         self.data = pd.DataFrame(
#             [
#                 ["Cat A", "Slice C", 0.1, 5, 300],
#                 ["Cat C", "Slice A", 0.2, 10, 3],
#                 ["Cat A", "Slice A", 0.3, 15, 5000],
#                 ["Cat B", "Slice B", 0.4, 20, 812],
#                 ["Cat B", "Slice D", 0.5, 25, 13312],
#             ]
#         )
#         self.model_name = "BERT"
#         self.dataset_name = "SNLI"
#
#     def test_init(self):
#         # Create a basic report
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#         )
#         self.assertTrue(self.data.equals(report.data))
#
#         # Pass config params
#         custom_color_scheme = ["#000000"]
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#             color_scheme=custom_color_scheme,
#         )
#         self.assertEqual(custom_color_scheme, report.config["color_scheme"])
#
#     def test_sort(self):
#         # Sort alphabetically
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#         )
#         report.sort()
#         actual = report.data
#         expected = pd.DataFrame(
#             [
#                 ["Cat A", "Slice A", 0.3, 15, 5000],
#                 ["Cat A", "Slice C", 0.1, 5, 300],
#                 ["Cat B", "Slice B", 0.4, 20, 812],
#                 ["Cat B", "Slice D", 0.5, 25, 13312],
#                 ["Cat C", "Slice A", 0.2, 10, 3],
#             ]
#         )
#         self.assertTrue(actual.equals(expected))
#
#         # Sort by specified category order
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#         )
#         report.sort(
#             category_order={
#                 "Cat B": 0,
#                 "Cat C": 2,
#                 "Cat A": 1,
#             }
#         )
#         actual = report.data
#         expected = pd.DataFrame(
#             [
#                 ["Cat B", "Slice B", 0.4, 20, 812],
#                 ["Cat B", "Slice D", 0.5, 25, 13312],
#                 ["Cat A", "Slice A", 0.3, 15, 5000],
#                 ["Cat A", "Slice C", 0.1, 5, 300],
#                 ["Cat C", "Slice A", 0.2, 10, 3],
#             ]
#         )
#         self.assertTrue(actual.equals(expected))
#
#         # Sort by specified slice order
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#         )
#         report.sort(
#             slice_order={"Slice D": 0, "Slice C": 1, "Slice B": 2, "Slice A": 3}
#         )
#         actual = report.data
#         expected = pd.DataFrame(
#             [
#                 ["Cat A", "Slice C", 0.1, 5, 300],
#                 ["Cat A", "Slice A", 0.3, 15, 5000],
#                 ["Cat B", "Slice D", 0.5, 25, 13312],
#                 ["Cat B", "Slice B", 0.4, 20, 812],
#                 ["Cat C", "Slice A", 0.2, 10, 3],
#             ]
#         )
#         self.assertTrue(actual.equals(expected))
#
#         # Sort by specified category order and slice order
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#         )
#         report.sort(
#             category_order={
#                 "Cat B": 0,
#                 "Cat C": 2,
#                 "Cat A": 1,
#             },
#             slice_order={"Slice D": 0, "Slice C": 1, "Slice B": 2, "Slice A": 3},
#         )
#         actual = report.data
#         expected = pd.DataFrame(
#             [
#                 ["Cat B", "Slice D", 0.5, 25, 13312],
#                 ["Cat B", "Slice B", 0.4, 20, 812],
#                 ["Cat A", "Slice C", 0.1, 5, 300],
#                 ["Cat A", "Slice A", 0.3, 15, 5000],
#                 ["Cat C", "Slice A", 0.2, 10, 3],
#             ]
#         )
#         self.assertTrue(actual.equals(expected))
#
#     def test_filter(self):
#         # Filter by category
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#         )
#         report.filter(categories=["Cat B"])
#         actual = report.data
#         expected = pd.DataFrame(
#             [
#                 ["Cat B", "Slice B", 0.4, 20, 812],
#                 ["Cat B", "Slice D", 0.5, 25, 13312],
#             ]
#         )
#         self.assertTrue(actual.equals(expected))
#
#         # Filter by slice
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#         )
#         report.filter(slices=["Slice A", "Slice C"])
#         actual = report.data
#         expected = pd.DataFrame(
#             [
#                 ["Cat A", "Slice C", 0.1, 5, 300],
#                 ["Cat C", "Slice A", 0.2, 10, 3],
#                 ["Cat A", "Slice A", 0.3, 15, 5000],
#             ]
#         )
#         self.assertTrue(actual.equals(expected))
#
#     def test_rename(self):
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#         )
#         category_map = {"Cat C": "Cat D"}
#         slice_map = {"Slice A": "Slice D"}
#         report.rename(category_map=category_map, slice_map=slice_map)
#         actual = report.data
#         expected = pd.DataFrame(
#             [
#                 ["Cat A", "Slice C", 0.1, 5, 300],
#                 ["Cat D", "Slice D", 0.2, 10, 3],
#                 ["Cat A", "Slice D", 0.3, 15, 5000],
#                 ["Cat B", "Slice B", 0.4, 20, 812],
#                 ["Cat B", "Slice D", 0.5, 25, 13312],
#             ]
#         )
#         self.assertTrue(actual.equals(expected))
#
#     def test_set_range(self):
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#         )
#         report.set_range("f1", 0.1, 0.3)
#         for col in report.columns:
#             if col.title == "f1":
#                 self.assertEqual((col.min_val, col.max_val), (0.1, 0.3))
#
#     def test_figure(self):
#         report = BarChart(
#             self.data,
#             self.cols,
#             model_name=self.model_name,
#             dataset_name=self.dataset_name,
#         )
#
#         # Original unsorted data should cause an error
#         self.assertRaises(ValueError, report.figure)
#
#         # Sort should resolve that error
#         report.sort()
#         try:
#             report.figure()
#         except ValueError:
#             self.fail("report.figure() raised ValueError unexpectedly!")


class TestReportGenerator(unittest.TestCase):
    # @unittest.skip("Manual test")
    def test_plot(self):
        reporter = ReportGenerator()
        reporter.plot(test_evaluate_result)


if __name__ == "__main__":
    unittest.main()
