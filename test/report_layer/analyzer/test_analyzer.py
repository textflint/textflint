import unittest

from textflint.report_layer.analyzer import Analyzer


evaluate_result = {
    "model_name": "BERT",
    "dataset_name": "medical data",
    "transformation": {
        "Case": {
            "ori_precision": 0.70,
            "trans_precision": 0.65,
            "ori_f1": 0.63,
            "trans_f1": 0.60,
            "size": 5000,
        },
        "Ocr": {
            "ori_precision": 0.72,
            "trans_precision": 0.43,
            "ori_f1": 0.62,
            "trans_f1": 0.41,
            "size": 5000,
        }
    },
    "subpopulation": {
        "LengthLengthSubPopulation-0.0-0.1": {
            "trans_precision": 0.68,
            "trans_f1": 0.63,
            "size": 500
        },
        "LMSubPopulation-0.0-0.2": {
            "trans_precision": 0.55,
            "trans_f1": 0.41,
            "size": 1000,
        }
    },
    "attack": {
        "Bert-Attack": {
            "ori_precision": 0.72,
            "trans_precision": 0.43,
            "ori_f1": 0.62,
            "trans_f1": 0.41,
            "size": 400,
        }
    }
}


class TestAnalyzer(unittest.TestCase):
    def test_json_to_bar_chart(self):
        Analyzer.json_to_bar_chart(evaluate_result)

    def test_json_to_sunburst(self):
        Analyzer.json_to_sunburst(evaluate_result)
        Analyzer.json_to_sunburst(evaluate_result, 'f1')
        self.assertRaises(ValueError, Analyzer.json_to_sunburst, {})

    def test_json_to_linguistic_radar(self):
        Analyzer.json_to_linguistic_radar(evaluate_result)
        self.assertRaises(ValueError, Analyzer.json_to_linguistic_radar, {})


if __name__ == '__main__':
    unittest.main()
