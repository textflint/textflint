"""
TextFlint Engine Class
============================================

"""

import os

from .common import logger
from .adapter import Adapter

__all__ = ["Engine"]


class Engine:
    r"""
    Engine class of Text Robustness.

    Support run entrance which automatically finish data loading,
    transformation/subpopulation/attack generation and robustness
    report generation.

    Also provide interfaces of each layer to practitioners.

    """

    def __init__(
        self,
        task='UT'
    ):
        r"""
        :param string task: supported task name
        """
        self.task = task

    def run(self, data_input, out_dir, config=None, model=None):
        r"""
        Engine start entrance, load data and apply transformations,
        finally generate robustness report if needed.

        :param dict|list|string data_input: json object or json/csv file
        :param string out_dir: out dir for saving generated samples
        :param string|TextFlint.Config config: json file or Config object
        :param TextFlint.FlintModel model: model wrapper which implements
            FlintModel abstract methods, not a necessary input.
        :return: save generated data to out dir
            and provide report in html format.

        """
        dataset, config, model = self.load(data_input, config, model)

        evaluate_result = self.generate(dataset, config, out_dir, model)

        if evaluate_result:
            self.report(evaluate_result)

    def load(self, data_input, config=None, model=None):
        r"""
        Load data input, config file and FlintModel.

        :param dict|list|string data_input: json object or json/csv file
        :param string|TextFlint.Config config: json file or Config object
        :param TextFlint.FlintModel model: model wrapper which implements
            FlintModel abstract methods, not a necessary input.
        :return: TextFlint.Dataset, TextFlint.Config, TextFlint.FlintModel

        """
        dataset = Adapter.get_dataset(
            data_input=data_input,
            task=self.task
        )

        config = Adapter.get_config(
            task=self.task,
            config=config
        )
        if model:
            model = Adapter.get_flintmodel(
                model=model,
                task=self.task
            )

        return dataset, config, model

    def generate(self, dataset, config, out_dir, model=None):
        r"""
        Generate new samples according to given config,
        save result as json file to out path, and evaluate
        model performance automatically if provide model.

        :param TextFlint.Dataset dataset: container of original samples.
        :param TextFlint.Config config: config instance to control procedure.
        :param str out_dir: out dir for saving generated samples
        :param TextFlint.FlintModel model: model wrapper which implements
            FlintModel abstract methods, not a necessary input.
        :return: save generated samples to json file.

        """
        generator = Adapter.get_generator(config)
        if model:
            model = Adapter.get_flintmodel(model, generator.task)
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
                    eval_json[trans_type].update(
                        model.evaluate(original_samples.dump(), prefix="ori_")
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
            report_generator = Adapter.get_report_generator()
            report_generator.plot(evaluate_result)
