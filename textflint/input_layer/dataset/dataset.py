r"""

dataset: textflint dataset
=============================
"""

from tqdm import tqdm

from ...common.utils import logger
from ..component.sample import Sample
from ...common.utils.load import task_class_load
from ...common.settings import SAMPLE_PATH, NLP_TASK_MAP
from ...common.utils.file_io import read_csv, read_json, save_csv, save_json


def get_sample_map():
    return task_class_load(SAMPLE_PATH,
                           [key.upper() for key in NLP_TASK_MAP.keys()],
                           Sample,
                           filter_str='_sample')


sample_map = get_sample_map()


class Dataset:
    r"""
    Any iterable of (label, text_input) pairs qualifies as a ``Dataset``.

    """
    def __init__(
        self,
        task='UT'
    ):
        r"""
        :param str task: indicate data sample format.

        """
        self._i = 0
        self.dataset = []

        if task.upper() not in sample_map:
            logger.warning(
                f'Do not support task: {task}, default utilize UT sample.')
            self.task = 'UT'
        else:
            self.task = task.upper()

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self.dataset):
            self.init_iter()
            raise StopIteration

        example = self.dataset[self._i]
        self._i += 1

        return example

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        if isinstance(i, int) or isinstance(i, slice):
            # `i` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return self.dataset[i]

    def new_dataset(self):
        return self.__class__(task=self.task)

    def init_iter(self):
        self._i = 0

    def free(self):
        r"""
        Fully clear dataset.

        """
        self._i = 0
        self.dataset = []

    def dump(self):
        r"""
        Return dataset in json object format.

        """
        json_samples = []
        for sample in self.dataset:
            try:
                json_samples.append(sample.dump())
            except ValueError as e:
                logger.error(str(e))
                logger.error('Skip {0} for failed dump.'.format(sample))
        return json_samples

    def load(self, dataset):
        r"""
        Loads json object and prepares it as a Dataset.

        Support two formats input,
        Example::

        1. {'x': [
                      'The robustness of deep neural networks has received
                      much attention recently',
                      'We focus on certified robustness of smoothed classifiers
                      in this work',
                      ...,
                      'our approach exceeds the state-of-the-art.'
                      ],
                'y': [
                      'neural',
                      'positive',
                      ...,
                      'positive'
                      ]}
            2. [
                {'x': 'The robustness of deep neural networks has received
                much attention recently', 'y': 'neural'},
                {'x': 'We focus on certified robustness of smoothed classifiers
                in this work', 'y': 'positive'},
                ...,
                {'x': 'our approach exceeds the state-of-the-art.',
                'y': 'positive'}
                ]
        :param list|dict dataset:
        :return:

        """
        sample_id = 0

        if isinstance(dataset, (Sample, list, dict)):
            logger.info('******Start load!******')

            success_count = 0
            norm_samples = self.norm_input(dataset)

            for single_sample in tqdm(norm_samples):
                success_count += self.append(single_sample, sample_id)
                sample_id += 1

            logger.info(
                '{0} in total, {1} were loaded successful.'.format(
                    len(norm_samples), success_count))
            logger.info('******Finish load!******')
        else:
            raise ValueError(
                'Cant load dataset type {0}'.format(
                    type(dataset)))

    def load_json(self, json_path, encoding='utf-8', fields=None, dropna=True):
        r"""
        Loads json file, each line of the file is a json string.

        :param json_path: file path
        :param encoding: file's encoding, default: utf-8
        :param fields: json object's fields that needed, if None,
            all fields are needed. default: None
        :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
        :return:

        """
        json_dics = []
        for line, json_dic in read_json(
                json_path, encoding=encoding, fields=fields, dropna=dropna):
            json_dics.append(json_dic)

        self.load(json_dics)

    def load_csv(
            self,
            csv_path,
            encoding='utf-8',
            headers=None,
            sep=',',
            dropna=True):
        r"""
        Loads csv file, one line correspond one sample.

        :param csv_path: file path
        :param encoding: file's encoding, default: utf-8
        :param headers: file's headers, if None, make file's first line
            as headers. default: None
        :param sep: separator for each column. default: ','
        :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
        :return:

        """
        json_dics = []
        for line, json_dic in read_csv(
                csv_path, encoding=encoding, headers=headers,
                sep=sep, dropna=dropna):
            json_dics.append(json_dic)

        self.load(json_dics)

    def load_hugging_face(self, name, subset="train"):
        r"""
        Loads a dataset from HuggingFace ``datasets``
        and prepares it as a Dataset.

        :param name: the dataset name
        :param subset: the subset of the main dataset.
        :return:

        """
        raise NotImplementedError

    def append(self, data_sample, sample_id=-1):
        r"""
        Load single data sample and append to dataset.

        :param dict|sample data_sample:
        :param int sample_id: useful to identify sample, default -1
        :return: True / False indicate whether append action successful.

        """
        load_success = False
        # default Sample input with sample_id
        if isinstance(data_sample, Sample):
            # different type would raise error
            if self.task.lower() not in data_sample.__repr__().lower():
                logger.error(
                    'Input data sample type {0} is not compatible with task {1}'
                    .format(data_sample.__repr__(), self.task))
            else:

                try:
                    # invalid data sample will filed in dump step
                    data_sample.dump()
                    self.dataset.append(data_sample)
                    load_success = True
                except ValueError as e:
                    logger.error(str(e))
                    logger.error(
                        'Invalid data sample {0} cuz its failed dump.'
                        .format(data_sample))
        elif isinstance(data_sample, dict):
            try:
                self.dataset.append(sample_map[self.task](
                    data_sample, sample_id=sample_id))
                load_success = True
            except (ValueError, AssertionError) as e:
                logger.error(str(e))
                logger.error('Data input {0} load failed, skip this sample'
                             .format(data_sample, self.task))
        else:
            logger.error('Not support append {0} type data to dataset, '
                         'check the input '.format(type(data_sample)))

        return load_success

    def extend(self, data_samples):
        r"""
        Load multi data samples and extend to dataset.

        :param list|dict|Sample data_samples:
        :return:

        """
        success_count = 0
        norm_samples = self.norm_input(data_samples)

        for single_data in norm_samples:
            if self.append(single_data):
                success_count += 1

        return len(norm_samples), success_count

    @staticmethod
    def norm_input(data_samples):
        r"""
        Convert various data input to list of dict.
        Example::

             {'x': [
                      'The robustness of deep neural networks has received
                      much attention recently',
                      'We focus on certified robustness of smoothed classifiers
                      in this work',
                      ...,
                      'our approach exceeds the state-of-the-art.'
                  ],
             'y': [
                      'neural',
                      'positive',
                      ...,
                      'positive'
                  ]
            }
            convert to
            [
                {'x': 'The robustness of deep neural networks has received
                much attention recently', 'y': 'neural'},
                {'x': 'We focus on certified robustness of smoothed classifiers
                in this work', 'y': 'positive'},
                ...,
                {'x': 'our approach exceeds the state-of-the-art.',
                'y': 'positive'}
            ]
        :param list|dict|Sample data_samples:
        :return: Normalized data.

        """
        if isinstance(data_samples, list):
            norm_samples = data_samples
        elif isinstance(data_samples, dict):
            keys = list(data_samples.keys())
            data_size = len(data_samples[keys[0]])
            norm_samples = []

            for key in keys:
                assert len(
                    data_samples[key]) == data_size, \
                    'Unmatch key length, {0} is {1}, while {2} is {3}!'.format(
                    keys[0], data_size, key, len(
                        data_samples[key]))

            for i in range(data_size):
                norm_samples.append(
                    dict([(key, data_samples[key][i]) for key in keys]))
        elif isinstance(data_samples, Sample):
            norm_samples = [data_samples]
        else:
            raise ValueError(
                'Data from pass is not instance of json, '
                'please check your data type:{0}'.format(
                    type(data_samples)))

        return norm_samples

    def save_csv(self, out_path, encoding='utf-8', headers=None, sep=','):
        r"""
        Save dataset to csv file.

        :param out_path: file path
        :param encoding: file's encoding, default: utf-8
        :param headers: file's headers, if None, make file's first line
            as headers. default: None
        :param sep: separator for each column. default: ','
        :return:

        """
        save_csv(
            self.dump(),
            out_path,
            encoding=encoding,
            headers=headers,
            sep=sep)
        logger.info('Save samples to {0}!'.format(out_path))

    def save_json(self, out_path, encoding='utf-8', fields=None):
        r"""
        Save dataset to json file which contains json object in each line.

        :param out_path: file path
        :param encoding: file's encoding, default: utf-8
        :param fields: json object's fields that needed, if None,
            all fields are needed. default: None
        :return:

        """
        save_json(self.dump(), out_path, encoding=encoding, fields=fields)
        logger.info('Save samples to {0}!'.format(out_path))
