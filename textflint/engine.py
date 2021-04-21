"""
textflint Engine Class
============================================

"""

import os

from .common import logger
from .adapter import *


__all__ = ["Engine"]


class Engine:
    r"""
    Engine class of Text Robustness.

    Support run entrance which automatically finish data loading,
    transformation/subpopulation/attack generation and robustness
    report generation.

    Also provide interfaces of each layer to practitioners.

    """

    def run(self, data_input, config=None, task=None, model=None):
        r"""
        Engine start entrance, load data and apply transformations,
        finally generate robustness report if needed.

        :param dict|list|string data_input: json object or json/csv file
        :param string|textflint.Config config: json file or Config object
        :param str task: task name which will be helpful without config input.
        :param textflint.FlintModel model: model wrapper which implements
            FlintModel abstract methods, not a necessary input.
        :return: save generated data to out dir
            and provide report in html format.

        """
        dataset, config, model = self.load(data_input, config, task, model)

        if len(dataset) == 0:
            raise ValueError("Empty dataset, please check your data format!")

        evaluate_result = self.generate(dataset, config, model)

        if evaluate_result:
            self.report(evaluate_result)

    def load(self, data_input, config=None, task=None, model=None):
        r"""
        Load data input, config file and FlintModel.

        :param dict|list|string data_input: json object or json/csv file
        :param string|textflint.Config config: json file or Config object
        :param str task: task name which will be helpful without config input.
        :param textflint.FlintModel model: model wrapper which implements
            FlintModel abstract methods, not a necessary input.
        :return: textflint.Dataset, textflint.Config, textflint.FlintModel

        """
        config = auto_config(config=config, task=task)

        dataset = auto_dataset(data_input=data_input, task=config.task)
        # Prefer to use the model passed from parameter
        model = model if model else config.flint_model
        model = auto_flintmodel(model=model, task=config.task)

        return dataset, config, model

    def generate(self, dataset, config, model=None):
        r"""
        Generate new samples according to given config,
        save result as json file to out path, and evaluate
        model performance automatically if provide model.

        :param textflint.Dataset dataset: container of original samples.
        :param textflint.Config config: config instance to control procedure.
        :param textflint.FlintModel model: model wrapper which implements
            FlintModel abstract methods, not a necessary input.
        :return: save generated samples to json file.

        """
        generator = auto_generator(config)
        out_dir = config.out_dir

        if model:
            model = auto_flintmodel(model, config.task)
        evaluate_result = {}
        generate_map = {
            "transformation": generator.generate_by_transformations,
            "subpopulation": generator.generate_by_subpopulations,
            "attack": generator.generate_by_attacks,
        }

        for generate_type in generate_map:
            eval_json = {}
            for original_samples, trans_samples, trans_type in \
                    generate_map[generate_type](dataset, model=model):
                out_suffix = '_' + trans_type + '_' +\
                             str(len(trans_samples)) + '.json'

                if original_samples:
                    logger.info(
                        f"{trans_type}, original {len(dataset)} samples, "
                        f"transform {len(original_samples)} samples!"
                    )
                    original_samples.save_json(
                        os.path.join(out_dir, 'ori' + out_suffix)
                    )
                else:
                    logger.info(
                        f"{trans_type}, collect {len(trans_samples)} samples"
                    )

                if trans_samples:
                    trans_samples.save_json(
                        os.path.join(out_dir, 'trans' + out_suffix)
                    )

                if model is not None and len(trans_samples):
                    eval_json[trans_type] = {"size": len(trans_samples)}
                    if original_samples:
                        eval_json[trans_type].update(
                            model.evaluate(
                                original_samples.dump(),
                                prefix="ori_"
                            )
                        )
                    eval_json[trans_type].update(
                        model.evaluate(trans_samples.dump(), prefix="trans_")
                    )
            if eval_json:
                evaluate_result[generate_type] = eval_json

        return evaluate_result

    def report(self, evaluate_result):
        r"""
        Automatically analyze the model robustness verification results
        and plot the robustness evaluation report.

        :param dict evaluate_result: json object contains robustness
            evaluation result and other additional information.
        :return: open a html of robustness report.

        """
        if evaluate_result:
            report_generator = auto_report_generator()
            report_generator.plot(evaluate_result)
