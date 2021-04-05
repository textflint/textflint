r"""
File io module, support csv and json.

============================================

"""

import os
import json
import csv
__all__ = ['read_csv', 'save_json', 'read_json', 'save_csv']


def read_csv(path, encoding='utf-8', headers=None, sep=',', dropna=True):
    r"""
    Construct a generator to read csv items.

    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param headers: file's headers, if None, make file's first line as headers.
        default: None
    :param sep: separator for each column. default: ','
    :param dropna: whether to ignore and drop invalid data,
        if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, csv item)
    """

    with open(path, 'r', encoding=encoding) as csv_file:
        f = csv.reader(csv_file, delimiter=sep)
        start_idx = 0
        if headers is None:
            headers = next(f)
            start_idx += 1
        elif not isinstance(headers, (list, tuple)):
            raise TypeError(
                "headers should be list or tuple, not {0}."
                    .format(type(headers)))

        for line_idx, line in enumerate(f, start_idx):
            contents = line
            if len(contents) != len(headers):
                if dropna:
                    continue
                else:
                    if "" in headers:
                        raise ValueError(
                            ("Line {0} has {1} parts, while header has "
                             "{2} parts.\nPlease check the empty parts "
                             "or unnecessary '{3}'s  in header.")
                            .format(line_idx, len(contents), len(headers), sep))
                    else:
                        raise ValueError(
                            "Line {0} has {1} parts, "
                            "while header has {2} parts."
                            .format(line_idx, len(contents), len(headers)))
            _dict = {}
            for header, content in zip(headers, contents):
                _dict[header] = content

            yield line_idx, _dict


def read_json(path, encoding='utf-8', fields=None, dropna=True):
    r"""
    Construct a generator to read json items.

    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param fields: json object's fields that needed, if None,
        all fields are needed. default: None
    :param dropna: whether to ignore and drop invalid data,
        if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, json item)
    """

    if fields:
        fields = set(fields)
    with open(path, 'r', encoding=encoding) as f:
        for line_idx, line in enumerate(f):
            data = json.loads(line)
            if fields is None:
                yield line_idx, data
                continue
            _res = {}
            for k, v in data.items():
                if k in fields:
                    _res[k] = v
            if len(_res) < len(fields):
                if dropna:
                    continue
                else:
                    raise ValueError(
                        'invalid instance at line: {0}'.format(line_idx))
            yield line_idx, _res


def save_csv(json_list, out_path, encoding='utf-8', headers=None, sep=','):
    r"""
    Save json list to csv file.

    :param json_list: list of dict
    :param out_path: file path
    :param encoding: file's encoding, default: utf-8
    :param headers: file's headers, if None, make file's first line as headers.
        default: None
    :param sep: separator for each column. default: ','
    :return:
    """
    if not json_list or not isinstance(json_list, list):
        raise ValueError(
            f'Cant save invalid data {json_list}, provide list of dict plz!')
    # mkdir dir automatically
    dir_path, file_path = os.path.split(out_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    headers = headers if headers else json_list[0].keys()

    with open(out_path, 'w+', encoding=encoding) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers, delimiter=sep)
        writer.writeheader()
        writer.writerows(json_list)


def save_json(json_list, out_path, encoding='utf-8', fields=None):
    r"""
    Save json list to json file which contains json object in each line.

    :param json_list: list of dict
    :param out_path: output path
    :param encoding: file's encoding, default: utf-8
    :param fields: json object's fields that needed, if None,
        all fields are needed. default: None
    :return:
    """

    if not json_list or not isinstance(json_list, list):
        raise ValueError(
            f'Cant save invalid data {json_list}, provide list of dict plz!')
    # mkdir dir automatically
    dir_path, file_path = os.path.split(out_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fields = fields if fields else json_list[0].keys()

    with open(out_path, 'w+', encoding=encoding) as json_file:
        for json_obj in json_list:
            out_json = {k: json_obj[k] for k in fields}
            json.dump(out_json, json_file, ensure_ascii=False)
            json_file.write('\n')


