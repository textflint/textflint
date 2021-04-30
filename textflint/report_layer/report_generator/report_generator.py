from __future__ import annotations

import itertools
from functools import partial
from typing import Dict, List

import dill
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from ..analyzer.analyzer import *


__all__ = ['ReportGenerator', 'BarChart']


class BarChart:
    r"""
    Class for textflint bar chart Report.
    Code from https://github.com/robustness-gym/robustness-gym

    """
    def __init__(
        self,
        data,
        columns,
        model_name,
        dataset_name,
        **kwargs,
    ):
        """

        :param pandas.DataFrame data: Pandas dataframe in the following format:
            column 1: category name
            column 2: slice name
            columns 3-N: data corresponding to passed columns parameter
        :param list[ReportColumn] columns: ReportColumn objects specifying
            format of columns 3-N in data
        :param str model_name: model name to show in report
        :param dataset_name: dataset name to show in report
        :param kwargs: any additional config paramters
        """

        # Make a copy of data since may be modified by methods below
        self.data = data.copy()

        self.columns = columns
        self.model_name = model_name
        self.dataset_name = dataset_name

        self.config = {
            "color_scheme": ["#ec7734", "#3499ec", "#ec34c1", "#9cec34"],
            "score_color_complement": "#F3F4F7",
            "text_fill_color": "#F3F4F7",
            "text_border_color": "#BEC4CE",
            "distribution_color_scale": [[0.0, "#FBF5F2"], [1.0, "#EC7734"]],
            "col_spacing": 0.035,
            "row_height": 24,
            "category_padding": 60,
            "header_padding": 80,
            "score_col_width": 0.6,
            "numeric_col_width": 0.25,
            "layout_width": 960,
            "font_size_dist": 12,
            "font_size_data": 13,
            "font_size_heading": 14,
            "font_size_category": 14,
        }

        self.update_config(**kwargs)

    def sort(self, category_order=None, slice_order=None):
        r"""
        Sort rows in report by category / slice alphabetically, or using
        specified order.

        :param category_order: map from category name to sorting rank. If None,
          sort categories alphabetically.
        :param slice_order: map from slice name to sorting rank. If None, sort
          slices alphabetically (within a category).

        """
        if category_order is None:
            category_order = {}

        if slice_order is None:
            slice_order = {}

        for col_name in ["sort-order-category", "sort-order-slice"]:
            if col_name in self.data:
                raise ValueError(f"Column name '{col_name}' is reserved")

        self.data["sort-order-category"] = self.data[0].map(
            lambda x: (category_order.get(x, 2 ** 10000), x)
        )
        self.data["sort-order-slice"] = self.data[1].map(
            lambda x: (slice_order.get(x, 2 ** 10000), x)
        )

        self.data = self.data.sort_values(
            by=["sort-order-category", "sort-order-slice"]
        ).drop(["sort-order-category", "sort-order-slice"], axis="columns")

        self.data.reset_index(inplace=True, drop=True)

    def filter(self, categories: List[str] = None, slices: List[str] = None):
        r"""
        Filter report to specific categories AND slices.

        :param List[str] categories: list of category names to filter by
        :param List[str] slices: list of slice names to filter by

        """
        if categories is not None:
            # self.data = self.data.loc(self.data[0].isin(categories))
            self.data = self.data[self.data[0].isin(categories)]
        if slices is not None:
            self.data = self.data[self.data[1].isin(slices)]
        self.data.reset_index(inplace=True, drop=True)

    def rename(self, category_map, slice_map):
        r"""
        Rename categories, slices

        :param Dict[str, str] category_map: map from old to new category name
        :param Dict[str, str] slice_map: map from old to new slice name

        """
        if category_map is not None:
            self.data[0] = self.data[0].map(lambda x: category_map.get(x, x))
        if slice_map is not None:
            self.data[1] = self.data[1].map(lambda x: slice_map.get(x, x))

    def set_model_name(self, model_name):
        """
        Set model name displayed on report.

        """
        self.model_name = model_name

    def set_dataset_name(self, dataset_name):
        """
        Set dataset name displayed on report.

        """
        self.dataset_name = dataset_name

    def set_range(self, col_title, min_val=None, max_val=None):
        """
        Set min and max values for score columns

        :param str col_title: title of column to update
        :param float min_val: minimum value
        :param float max_val: maximum value

        """
        for col in self.columns:
            if isinstance(col, ScoreColumn) and col.title == col_title:
                if min_val is not None:
                    col.min_val = min_val
                if max_val is not None:
                    col.max_val = max_val

    def update_config(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.config:
                raise ValueError(f"Invalid config param: '{k}'")
            self.config[k] = v

    def round(self):
        # Round everything
        self.data = self.data.round(3)
        self.data.pred_dist = self.data.pred_dist.apply(
            partial(np.round, decimals=3)
        )

    @classmethod
    def load(cls, path):
        obj = dill.load(open(path, "rb"))
        assert isinstance(obj, BarChart), (
            f"dill loaded an instance of {type(obj)}, "
            f"" f"must load {cls.__name__}."
        )
        return obj

    def save(self, path):
        return dill.dump(self, open(path, "wb"))

    def figure(self, show_title=False):
        # Verify that rows are grouped by category
        row_categories = self.data[0].tolist()
        # Previous category groupings already encountered
        save_cat_groups = set()
        prev_cat = None
        # Loop through each row and see
        # if a category is encountered outside of first
        # identified group for that category
        for cat in row_categories:
            if cat != prev_cat:  # category changes
                # if new category previously encountered
                if cat in save_cat_groups:
                    raise ValueError("Rows must be grouped by category.")
                prev_cat = cat
                save_cat_groups.add(cat)

        categories = []
        category_sizes = []  # Num rows in each category
        # column 0 is category
        for category, group in itertools.groupby(self.data[0]):
            categories.append(category)
            category_sizes.append(len(list(group)))
        n_rows = sum(category_sizes)
        height = (
            n_rows * self.config["row_height"]
            + len(categories) * self.config["category_padding"]
            + self.config["header_padding"]
        )
        col_widths = []
        for col in self.columns:
            if isinstance(col, ScoreColumn):
                col_width = self.config["score_col_width"]
            else:
                col_width = self.config["numeric_col_width"]
            col_widths.append(col_width)

        fig = make_subplots(
            rows=len(categories),
            row_titles=categories,
            cols=len(self.columns),
            shared_yaxes=True,
            subplot_titles=[col.title for col in self.columns],
            horizontal_spacing=self.config["col_spacing"],
            vertical_spacing=self.config["category_padding"] / height,
            row_width=list(reversed(category_sizes)),
            column_width=col_widths,
        )

        hms = []
        coords = []
        category_ndx = 1
        # Group data by category
        for category, category_data in self.data.groupby(0, sort=False):
            score_col_ndx = 0
            slice_names = category_data[1]
            slice_names = [s + " " * 3 for s in slice_names]
            for col_ndx, col in enumerate(self.columns):
                df_col_ndx = col_ndx + 2
                # Dataframe has two leading columns with category, slice
                fig_col_ndx = col_ndx + 1  # figure columns are 1-indexed
                x = category_data[df_col_ndx].tolist()
                if isinstance(col, ScoreColumn):
                    if col.is_0_to_1:
                        x = [100 * x_i for x_i in x]
                    col_max = col.max_val
                    if col.is_0_to_1:
                        col_max = 100 * col.max_val
                    fig.add_trace(
                        go.Bar(
                            x=x,
                            y=slice_names,
                            orientation="h",
                            marker=dict(color=self.get_color(score_col_ndx)),
                            showlegend=False,
                            text=[f"{x_i:.1f}" for x_i in x],
                            textposition="inside",
                            width=0.95,
                            textfont=dict(color="white"),
                        ),
                        row=category_ndx,
                        col=fig_col_ndx,
                    )
                    # Add marker for gray fill
                    fig.add_trace(
                        go.Bar(
                            x=[col_max - x_i for x_i in x],
                            y=slice_names,
                            orientation="h",
                            marker=dict(
                                color=self.config["score_color_complement"]
                            ),
                            showlegend=False,
                            width=0.9,
                        ),
                        row=category_ndx,
                        col=fig_col_ndx,
                    )
                    score_col_ndx += 1
                elif isinstance(col, NumericColumn):
                    # Repurpose bar chart as text field.
                    fig.add_trace(
                        go.Bar(
                            x=[1] * len(x),
                            y=slice_names,
                            orientation="h",
                            marker=dict(
                                color=self.config["text_fill_color"],
                                line=dict(
                                    width=0,
                                    color=self.config["text_border_color"]
                                ),
                            ),
                            showlegend=False,
                            text=[human_format(x_i) for x_i in x],
                            textposition="inside",
                            insidetextanchor="middle",
                            width=0.9,
                        ),
                        row=category_ndx,
                        col=fig_col_ndx,
                    )
                else:
                    raise ValueError("Invalid col type")
            category_ndx += 1

        for category_ndx in range(1, len(categories) + 1):
            if category_ndx == len(categories):
                show_x_axis = True
            else:
                show_x_axis = False
            for col_ndx, col in enumerate(self.columns):
                # plotly cols are 1-indexed
                fig_col_ndx = col_ndx + 1
                fig.update_yaxes(autorange="reversed", automargin=True)
                if isinstance(col, ScoreColumn):
                    if col.is_0_to_1:
                        col_min, col_max = 100 * col.min_val, 100 * col.max_val
                    else:
                        col_min, col_max = col.min_val, col.max_val

                    fig.update_xaxes(
                        range=[col_min, col_max],
                        row=category_ndx,
                        col=fig_col_ndx,
                        tickvals=[col_min, col_max],
                        showticklabels=show_x_axis,
                    )
                elif isinstance(col, NumericColumn):
                    fig.update_xaxes(
                        range=[0, 1],
                        row=category_ndx,
                        col=fig_col_ndx,
                        showticklabels=False,
                    )

        fig.update_layout(
            height=height,
            width=self.config["layout_width"],
            barmode="stack",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(size=self.config["font_size_data"]),
            yaxis={"autorange": "reversed"},
            margin=go.layout.Margin(
                # right margin  # bottom margin  # top margin
                r=0, b=0, t=20
            ),
        )

        # Use low-level plotly interface to update padding / font size
        for a in fig["layout"]["annotations"]:
            # If label for group
            if a["text"] in categories:
                a["x"] = 0.99  # Add padding
                a["font"] = dict(size=self.config["font_size_category"])
            else:
                a["font"] = dict(
                    size=self.config["font_size_heading"]
                )  # Adjust font size for non-category labels

        # Due to a quirk in plotly, need to do some special low-level coding
        # Code from https://community.plotly.com/t/how-to-create-
        # annotated-heatmaps-in-subplots/36686/25
        newfont = [
            go.layout.Annotation(font_size=self.config["font_size_heading"])
        ] * len(fig.layout.annotations)
        fig_annots = [newfont] + [hm.layout.annotations for hm in hms]
        for col_ndx in range(1, len(fig_annots)):
            for k in range(len(fig_annots[col_ndx])):
                coord = coords[col_ndx - 1]
                fig_annots[col_ndx][k]["xref"] = f"x{coord}"
                fig_annots[col_ndx][k]["yref"] = f"y{coord}"
                fig_annots[col_ndx][k]["font_size"] = self.config[
                    "font_size_dist"
                ]

        def recursive_extend(mylist, nr):
            # mylist is a list of lists
            result = []
            if nr == 1:
                result.extend(mylist[nr - 1])
            else:
                result.extend(mylist[nr - 1])
                result.extend(recursive_extend(mylist, nr - 1))
            return result

        new_annotations = recursive_extend(fig_annots[::-1], len(fig_annots))
        fig.update_layout(annotations=new_annotations)

        if show_title:
            title = {
                "text": f"{self.dataset_name or ''} {self.model_name or ''} "
                f"Robustness Report",
                "x": 0.5,
                "xanchor": "center",
            }
        else:
            title = None
        fig.update_layout(
            title=title,
            margin=go.layout.Margin(
                r=0, b=0, t=80  # right margin  # bottom margin  # top margin
            ),
        )

        return fig

    def get_color(self, col_ndx):
        return self.config["color_scheme"][col_ndx %
                                           len(self.config["color_scheme"])]


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."),
        ["", "K", "M", "B", "T"][magnitude]
    )


class ReportGenerator:
    r"""
    Plotting robustness report,
    return radar figure, sunbrust figure, and bar chart figure.

    Example:
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

    def plot(self, evaluate_json):
        r"""
        Analysis evaluation result and plot three reports in html.

        :param dict evaluate_json: evaluate result of specific model.

        """
        model_name = evaluate_json.get("model_name", None)
        dataset_name = evaluate_json.get("dataset_name", None)

        radar_pd = Analyzer.json_to_linguistic_radar(evaluate_json)
        if radar_pd is not None:
            self.get_radar_fig(radar_pd).show()

        df, settings = Analyzer.json_to_sunburst(evaluate_json)
        if df is not None:
            self.get_sunburst_fig(df, settings).show()

        pd, cols = Analyzer.json_to_bar_chart(evaluate_json)
        if pd is not None:
            self.get_bar_chart(pd, cols, model_name=model_name,
                               dataset_name=dataset_name).show()

    @staticmethod
    def get_radar_fig(radar_pd):
        r"""
        Get radar figure of linguistic classifications.

        """
        fig = px.line_polar(radar_pd, r='r', theta='theta',
                            line_close=True)
        fig.update_traces(fill='toself')

        return fig

    @staticmethod
    def get_sunburst_fig(df, settings):
        r"""
        Get sunburst figure of linguistic classifications and show details.

        """
        fig = px.sunburst(df, **settings)

        return fig

    @staticmethod
    def get_bar_chart(pd, cols, model_name=None, dataset_name=None):
        r"""
        Get bar chart figure.

        """
        fig = BarChart(
            pd,
            cols,
            model_name=model_name,
            dataset_name=dataset_name,
        )

        fig.sort()

        return fig.figure()
