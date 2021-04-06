import json
import os
import pkgutil
import importlib
import tarfile
from tempfile import NamedTemporaryFile
import time
import pandas as pd
import torch
import glob

from . import device
from .install import download_if_needed


def module_loader(dir_path, filter_str=''):
    assert os.path.exists(dir_path)

    for module in pkgutil.iter_modules([dir_path]):
        # filter illegal module
        if filter_str != '' and module.name.find(filter_str) == -1:
            continue
        module_path = os.path.join(dir_path, module.name)
        module_name = os.path.relpath(module_path).replace(os.path.sep, '.')
        assert('textflint' in module_name)
        module_name = module_name[module_name.find('textflint'):]
        yield importlib.import_module(module_name)
        # yield module.module_finder.find_loader(module.name)[0].load_module()


def task_class_load(pkg_path, task_list, base_class, filter_str=''):
    modules = module_loader(pkg_path, filter_str)
    task_class_map = {}
    for module in modules:
        task = module.__name__.split('.')[-1].split('_')[0]
        task_class = None
        assert task.upper() in task_list

        for attr in dir(module):
            reference = getattr(module, attr)
            if type(reference).__name__ not in ['classobj', 'ABCMeta']:
                continue
            if issubclass(reference, base_class) and reference != base_class:
                task_class = reference
                break

        if task_class is None:
            raise ImportError(
                'Not find task config in {0}, '
                'plz insure your implementation class extend base Class.'
                .format(module.name))
        task_class_map[task.upper()] = task_class

    return task_class_map


def pkg_class_load(pkg_path, base_class, filter_str=''):
    modules = module_loader(pkg_path, filter_str)
    subclasses = {}

    for module in modules:
        for attr in dir(module):
            reference = getattr(module, attr)
            if type(reference).__name__ not in ['classobj', 'ABCMeta']:
                continue
            if issubclass(reference, base_class) and reference != base_class:
                subclasses[attr] = reference

    return subclasses


def load_module_from_file(module_name, file_path):
    """Uses ``importlib`` to dynamically open a file and load an object from
    it."""
    try:
        temp_module_name = f"temp_{time.time()}"
        spec = importlib.util.spec_from_file_location(
            temp_module_name, file_path)
        file_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(file_module)
    except Exception as e:
        print(e)
        raise ValueError(f"Failed to import file {file_path}")
    try:
        module = getattr(file_module, module_name)
    except AttributeError:
        raise AttributeError(
            f"``{module_name}`` not found in module {file_path}"
        )
    return module


def load_cached_state_dict(model_folder_path):
    model_folder_path = download_if_needed(model_folder_path)
    # Take the first model matching the pattern *model.bin.
    model_path_list = glob.glob(os.path.join(model_folder_path, "*model.bin"))
    if not model_path_list:
        raise FileNotFoundError(
            f"model.bin not found in model folder {model_folder_path}."
        )
    model_path = model_path_list[0]
    state_dict = torch.load(model_path, map_location=device)
    return state_dict


def plain_lines_loader(path):
    """
        read data
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def json_loader(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def json_lines_loader(path):
    with open(path, 'r', encoding='UTF-8') as f:
        lines = [json.loads(line) for line in f.readlines()]
    return lines


# -------------------------SA load-------------------------
def sa_dict_loader(path):
    name_dict = {}
    info_csv = pd.read_csv(path,
                           names=['name', 'summary'])
    for row in info_csv.iterrows():
        name_dict[row[1]['name']] = str(row[1]['summary'])
    max_len = -1
    for item in name_dict.keys():
        item_len = len(item.split())
        if item_len > max_len:
            max_len = item_len
    return name_dict, max_len


# -------------------------ABSA load-------------------------
def absa_dataset_loader(path):
    return [value for key, value in json_loader(path).items()]


def absa_dict_loader(path):
    examples = json_loader(path)
    return examples


# -------------------------NER load-------------------------
def load_oov_entities(path):
    dic = {'PER': [], 'ORG': [], 'LOC': [], 'MISC': []}
    for line in plain_lines_loader(path):
        line = line.strip().split(' ')
        entity = line[0]
        for i in range(1, len(line) - 1):
            entity += ' ' + line[i]
        dic[line[len(line) - 1]] += [entity]
    return dic


def read_cross_entities(path):
    dic = {}
    for line in plain_lines_loader(path):
        line = line.strip().split(' ')
        label = line[-1:][0]
        word = line[0]
        line = line[1:-1]
        for i in line:
            word += ' ' + i
        if label not in dic:
            dic[label] = [word]
        else:
            dic[label].append(word)
    return dic
# -------------------------MRC load---------------------------


def load_supporting_file(neighbor_path, pos_path):
    with open(pos_path) as f:
        pos_tag_dict = json.load(f)
    with open(neighbor_path) as f:
        nearby_word_dict = json.load(f)
    return nearby_word_dict, pos_tag_dict

# -------------------------POS load-------------------------


def load_morfessor_model(path):
    import morfessor
    s = tarfile.open(path)
    file_handler = s.extractfile(s.next())
    tmp_file_ = NamedTemporaryFile(delete=False)
    tmp_file_.write(file_handler.read())
    tmp_file_.close()
    io = morfessor.MorfessorIO()
    model = io.read_any_model(tmp_file_.name)
    os.remove(tmp_file_.name)
    return model
