import pandas as pd
import numpy as np
from functools import reduce
from copy import deepcopy
from collections import OrderedDict

from ...common.utils import logger

__all__ = ['Analyzer', 'ReportColumn', 'ScoreColumn', 'NumericColumn']

CATEGORY = {
    "Morphology": [
        "SwapPrefix",
        "Tense",
        "SwapVerb",
        "SwapMultiPOS",
        "SwapAcronym",
        "SwapLonger",
        "SpellingError",
        "Keyboard",
        "Typos",
        "Ocr",
        "EntTypos"
    ],
    "Syntax": [
        "SwapNamedEnt",
        "OOV/CrossCategory",
        "SwapSpecialEnt",
        "SwapTriplePos",
        "DoubleDenial",
        "SwapWord",
        "SwapEnt",
        "RevTgt",
        "RevNon",
        "InsertAdv",
        "DeleteSubTree",
        "AddSubTree",
        "DeleteAdd",
        "InsertClause"
    ],
    "Pragmatics": [
        "RndShuffle",
        "Punctuation",
        "AddSum",
        "RndRepeat/Delete",
        "AddSent",
        "AddDiff",
        "AppendIrr",
        "TwitterType",
        "RndInsert",
        "ConcatSent",
        "AddSentDiverse",
        "PerturbAnswer/Question"
    ],
    "ParadigmaticRelation": [
        "SwapNum",
        "SwapSyn",
        "SwapContraction",
        "SwapAnt",
        "ReverseNeg",
        "SwapName",
        "MLMSuggestion"
    ],
    "Other": [
        "BackTrans",
        "Overlap",
        "ModifyPos"
    ]
}


class ReportColumn:
    """
    A single column in the Robustness Report.

    """

    def __init__(
        self,
        title
    ):
        self.title = title

    def set_title(self, title):
        self.title = title


class ScoreColumn(ReportColumn):
    """
    A column for numeric scores in the Robustness Report, displayed as a bar
    chart.

    """

    def __init__(
        self,
        title,
        min_val,
        max_val,
        is_0_to_1=False
    ):
        super(ScoreColumn, self).__init__(title)
        self.min_val = min_val
        self.max_val = max_val
        self.is_0_to_1 = is_0_to_1

    def set_min(self, min_val: float):
        self.min_val = min_val

    def set_max(self, max_val: float):
        self.max_val = max_val


class NumericColumn(ReportColumn):
    """
    A column for numeric data in the Robustness Report, displayed as the raw
    value.

    """

    def __init__(
        self,
        title
    ):
        super(NumericColumn, self).__init__(title)


class Analyzer:
    r"""
    Convert evaluate result json to DataFrame for report generator,
    and analysis model robustness according to linguistic classification.

    Example::

        {
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
    """

    @staticmethod
    def json_to_bar_chart(evaluate_json):
        r"""
        Parsing evaluate json and convert to bar chart input format.

        :param dict evaluate_json: evaluate result of specific model.
        :return: pandas.DataFrame, list[ReportColumn]

        """
        bar_json_list = []

        for generate_type in ["transformation", "subpopulation", "attack"]:
            generate_methods = evaluate_json.get(generate_type, {})
            if generate_methods:
                for method in generate_methods:
                    bar_json = OrderedDict({
                        "generate_type": generate_type,
                        "generate_method": method,
                    })
                    metrics = {
                        (k, v) for k, v in generate_methods[method].items()
                        if k != "size"
                    }
                    bar_json.update(metrics)

                    bar_json["size"] = generate_methods[method].get("size", 0)
                    bar_json_list.append(bar_json)

        if bar_json_list is []:
            return None, None

        df = pd.DataFrame.from_dict(bar_json_list, orient='columns')

        cols = [ScoreColumn(col, 0, 1, is_0_to_1=True)
                for col in df if col not in
                ["generate_method", "generate_type", "size"]
                ] + [NumericColumn("Size")]
        df.columns = range(len(df.columns))

        return df, cols

    @staticmethod
    def json_to_sunburst(evaluate_json, metric=None):
        r"""
        Parsing evaluate json and classify each transformation.

        :param dict evaluate_json: evaluate result of specific model.
        :param str metric: key metric to plot subburst figure.
        :return: pandas.DataFrame, dict

        """
        if "transformation" not in evaluate_json:
            logger.info(("Cant find transformation in given json, "
                        "skip sunburst report generation!"))
            return None, None

        transformations = evaluate_json["transformation"]
        metric = Analyzer.get_metric(transformations, metric)
        sunburst_list = []
        hover_data = None

        for transformation in transformations:
            trans_json = deepcopy(transformations[transformation])
            trans_json['transformation'] = transformation
            trans_json['parent'] = Analyzer.get_parent(transformation)
            sunburst_list.append(trans_json)
            if not hover_data:
                hover_data = [
                    key for key in transformations[transformation].keys()
                ]
        if metric in hover_data:
            hover_data.remove(metric)

        df = pd.DataFrame.from_dict(sunburst_list, orient='columns')

        sunburst_settings = {
            'path': ['parent', "transformation"],
            'color': metric,
            'values': 'size',
            'hover_data': hover_data,
            'color_continuous_scale': 'RdBu',
            'color_continuous_midpoint': np.average(df[metric])

        }

        return df, sunburst_settings

    @staticmethod
    def get_metric(transformations, metric=None):
        """
        Get key metric of given transformations.

        :param dict transformations: evaluation result of transformation
        :param str metric: key metric to plot subburst figure.
        :return: str legal metric name

        """
        if len(transformations) < 1:
            raise ValueError(f"Cant get metric of {transformations} to plot!")
        if metric:
            assert isinstance(metric, str), f"Cant recognize metric {metric}"
            metrics = [metric]
        else:
            metrics = []

        for transformation in transformations:
            original_result, transform_result = \
                Analyzer.get_metrics(transformations[transformation])
            if metrics:
                metrics = reduce(
                    lambda x, y: list(set(x).intersection(list(y))),
                    [metrics, original_result.keys(), transform_result.keys()]
                )
            else:
                metrics = list(original_result.keys())
        if len(metrics) < 1:
            raise ValueError(f"Failed to load metric value for "
                             f"{transformations}, cuz lack of metric scores.")
        else:
            return 'trans_' + metrics[0]

    @staticmethod
    def get_parent(transformation_str):
        """
        Find linguistic classification of given transformation,
        if not found, return Other label.

        :param str transformation_str: transformation name
        :return: str linguistic classification name

        """
        parent = "Other"

        for category_type in CATEGORY:
            if transformation_str in CATEGORY[category_type]:
                parent = category_type
                break

        return parent

    @staticmethod
    def json_to_linguistic_radar(evaluate_json):
        r"""
        Parsing evaluation result and calculate linguistic robustness scores.

        :param dict evaluate_json: evaluate result of specific model.
        :return: pandas.DataFrame

        """
        if "transformation" not in evaluate_json:
            logger.info(("Cant find transformation in given json, "
                         "skip linguistic radar report generation!"))
            return None

        transformations = evaluate_json['transformation']
        scores = {category_type: [] for category_type in CATEGORY}

        for transformation in transformations:
            score = Analyzer.radar_score(transformations[transformation])
            is_record = False
            # transformation not in current category is considered as other type
            for category_type in CATEGORY:
                if transformation in CATEGORY[category_type]:
                    scores[category_type].append(score)
                    is_record = True
                    break
            if not is_record:
                scores['Other'].append(score)

        for score in scores:
            if scores[score]:
                scores[score] = (1 - reduce(lambda x, y: x + y, scores[score])
                                 / len(scores[score])) * 5
            else:
                scores[score] = 5

        return pd.DataFrame(
            dict(
                r=list(scores.values()),
                theta=list(scores.keys())
            )
        )

    @staticmethod
    def radar_score(trans_json):
        """
        Get radar score by calculate average metric decreasing ratio.

        :param dict trans_json: evaluation result of specific
            transformation.
        :return: pandas.DataFrame

        """
        assert isinstance(trans_json, dict), \
            f"transformation evaluation should be dict type, " \
            f"given {type(trans_json)}"

        original_result, transform_result = Analyzer.get_metrics(trans_json)
        decreasing_ratio = []

        for metric in original_result:
            ori_score = float(original_result[metric])
            trans_score = float(transform_result[metric])
            decreasing_ratio.append(
                max(0, (ori_score - trans_score) / ori_score)
            )

        return reduce(lambda x, y: x + y, decreasing_ratio) \
               / len(decreasing_ratio)

    @staticmethod
    def get_metrics(trans_json):
        """
        Parsing and checking evaluation result of specific transformation.

        :param dict trans_json: evaluation result.
        :return: dict, dict

        """
        original_result = {}
        transform_result = {}

        for key in trans_json:
            if "ori_" in key:
                metric = key.split("ori_")[1]
                original_result[metric] = trans_json[key]
            elif "trans_" in key:
                metric = key.split("trans_")[1]
                transform_result[metric] = trans_json[key]

        assert set(list(original_result.keys())) == \
               set(list(transform_result.keys())), \
            f"Original metric {original_result.keys()} unmatch with " \
            f"transform metric {transform_result.keys()}"

        return original_result, transform_result
