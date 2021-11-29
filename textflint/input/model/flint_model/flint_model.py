from abc import ABC
import numpy as np

from ..metrics.metrics import accuracy_score as Accuracy
from ..metrics.metrics import POSMetric
from ..metrics.nerspanmetric import NERSpanMetric


__all__ = ["FlintModel", "TASK_METRICS"]

TASK_METRICS = {
    'SA': [{"name": "accuracy", "fun": Accuracy}],
    'POS': [{"name": "accuracy", "fun": POSMetric}],
    'CWS': [{"name": ["precision", "recall", "f1_socre"],
             "fun": NERSpanMetric}],
    'NER': [{"name": ["precision", "recall", "f1_socre"],
             "fun": NERSpanMetric}]
}

CLASSIFICATION_TASKS = ['ABSA', 'SA', 'SM', 'NLI', 'TC', 'POS']
ALLOWED_ATTACK_TASKS = ['SA', 'SM', 'NLI', 'TC']


class FlintModel(ABC):
    r"""
    A model wrapper queries a model with a list of text inputs.

    Classification-based models return a list of lists, where each sublist
    represents the model's scores for a given input.

    Text-to-text models return a list of strings, where each string is the
    output – like a translation or summarization – for a given input.

    """

    def __init__(
        self,
        model,
        tokenizer,
        task='SA',
        batch_size=1
    ):
        r"""

        :param model: any model object
        :param tokenizer: support tokenize sentence and convert tokens to
            model input ids
        :param str task: task name
        :param int batch_size: batch size to apply evaluation

        """
        if task not in TASK_METRICS:
            raise ValueError(f'Not support task {task} !')
        self.task = task
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def evaluate(self, data_samples, prefix=''):
        r"""
        :param list[Sample] data_samples: list of Samples
        :param str prefix: name prefix to add to metrics
        :return: dict obj to save metrics result

        """
        outputs = []
        labels = []
        i = 0

        while i < len(data_samples):
            batch_samples = data_samples[i: i + self.batch_size]
            batch_inputs, batch_labels = self.unzip_samples(batch_samples)
            labels += batch_labels
            predicts = self.__call__(*batch_inputs)

            if self.task in CLASSIFICATION_TASKS:
                predicts = np.argmax(predicts, axis=-1)
            outputs += predicts.tolist()
            i += self.batch_size

        metrics_rst = {}
        for Metric in TASK_METRICS[self.task]:
            metrics_rst[prefix + Metric["name"]] \
                = Metric["fun"](outputs, np.array(labels))

        return metrics_rst

    def get_grad(self, *inputs):
        r"""
        Get gradient of loss with respect to input tokens.

        :param tuple inputs: tuple of original texts

        """
        if self.task not in ALLOWED_ATTACK_TASKS:
            raise RuntimeError(f"Not support task {self.task} current...")

        return self.get_model_grad(*inputs)

    def __call__(self, *inputs):
        r"""
        Prepare model input ids and get model predict output.

        *inputs
            1、attack supported tasks(1/2 text string)
            2、other tasks(original sample content)

        Returns:
            1、attack supported tasks(label scores)
            2、other tasks(label ids)

        :param tuple inputs: list of original text
        :return: numpy.ndarray
        """
        raise NotImplementedError()

    def get_model_grad(self, *inputs):
        r"""
        Get gradient of loss with respect to input tokens.

        :param tuple inputs: list of original text

        """
        raise NotImplementedError()

    def unzip_samples(self, data_samples):
        r"""
        Unzip sample to input texts and labels.

        :param list data_samples: sample list
        :return: (inputs_text), labels.

        """
        raise NotImplementedError()
