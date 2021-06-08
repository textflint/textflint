import torch
from .flint_model import FlintModel, TASK_METRICS

__all__ = ['FlintModelCWS']


class FlintModelCWS(FlintModel):
    def __init__(self,
                 model,
                 batch_size=1,
                 metrics=None,
                 tag_vocab=['B', 'M', 'E', 'S'],
                 encoding_type='bmes'
                 ):
        r"""
        :param model: any model object
        :param int batch_size: batch size to apply evaluation
        :param Object metrics: metrics takes the predicted labels
        and the ground-truth labels as input and calculates f1, recall
        and precision. It takes a SpanFPreRecMetric metric
        as default, but can also be implemented by the user.
        :param list tag_vocab: a list of tag.
        for mapping tags to indexes.
        :param str encoding_type: supporting 'bio', 'bmes', 'bmeso', 'bioes',
        'bio' as default.
        """
        super(FlintModelCWS, self).__init__(
            model=model,
            tokenizer=None,
            task='CWS',
            batch_size=batch_size)
        # load augments for the model
        self.batch_size = batch_size
        self.model = model
        self.metrics = metrics
        self.tag = tag_vocab
        self.tag_vocab = self.get_vocab(tag_vocab)
        self.encoding_type = encoding_type
        self.metrics2score = {'f': 'f1_score',
                              'pre': 'precision', 'rec': 'recall'}

    def evaluate(self, data_samples, prefix=''):
        r"""
        :param list[Sample] data_samples: list of Samples
        :return: dict obj to save metrics result
        """

        dataset = self.encode(data_samples)
        data_iterator = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )

        if not self.metrics:
            self.metrics = TASK_METRICS[self.task][0]['fun'](
                encoding_type=self.encoding_type,
                tag_vocab=self.tag
            )

        with torch.no_grad():
            for batch_x, batch_y in data_iterator:
                pred = self.model(batch_x)  # [batch, seq_len]
                target = batch_y  # [batch, seq_len]
                seq_len = [torch.tensor(len(i)) for i in batch_x]
                self.metrics.evaluate(pred, target, torch.stack(seq_len))

        metric = self.metrics.get_metric()
        score = {}
        for i in metric.keys():
            score[prefix + self.metrics2score[i]] = metric[i]
        print(score)
        return score

    def encode(self, inputs):
        r"""
        :param list[dict] data_samples: list of dict data
        :return: DataSet obj obtaining

        """
        sentences = []
        label = []
        for sample in inputs:
            sentences.append(sample['x'])
            label.append([self.tag_vocab[i] for i in sample['y']])

        return sentences, label

    @staticmethod
    def collate_fn(batch):
        # change the batch data to tensor
        max_len = 0
        for i in batch[1]:
            max_len = max(max_len, len(i))
        return [i for i in batch[0]], \
               torch.tensor([i + [-1] * (max_len - len(i)) for i in batch[1]])

    def get_vocab(self, vocab):
        r"""
        :param list vocab: the list of the label
        :return: dict

        """
        assert isinstance(vocab, list)
        res = {}
        for i, j in enumerate(vocab):
            res[j] = i
        return res
