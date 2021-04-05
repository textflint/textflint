import os
import sys
import json
from copy import deepcopy
from itertools import product, combinations

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from TextFlint.common.config_files.default_settings import *


def get_allowed_ut_set(task):
    ut_set = TRANSFORMATIONS['UT']

    return [x for x in ut_set if x not in NOT_ALLOWED_UT_TRANS[task]]


def get_task_set(task):

    return TRANSFORMATIONS[task]


def gen_ut(task):

    return get_allowed_ut_set(task)


def gen_task(task):

    return get_task_set(task)


def gen_ut_task_com(task):
    combs = []
    task_set = get_task_set(task)
    ut_set = get_allowed_ut_set(task)
    ori_combs = [comb for comb in product(task_set, ut_set)]

    for comb in ori_combs:
        trans_priorities = [TRANSFORMATION_PRIORITY.get(trans, 0) for trans in comb]
        sorted_trans = sorted(zip(comb, trans_priorities), key=lambda x: x[1]) 
        combs.append([trans[0] for trans in sorted_trans])

    return combs


def gen_ut_ut_com(task):
    combs = []
    ut_set = get_allowed_ut_set(task)

    for comb in combinations(ut_set, 2):
        trans_priorities = [TRANSFORMATION_PRIORITY.get(trans) for trans in comb]
        sorted_trans = sorted(zip(comb, trans_priorities), key=lambda x: x[1]) 
        combs.append([trans[0] for trans in sorted_trans])

    return combs

# 一键生成 task单变形， UT单变形， task+UT, UT+UT 四种变形文件
def gen_configs(task):
    if task and not os.path.exists(task):
        os.makedirs(task)
    cnt = 0
    for config_type, gen_method in [('UT', gen_ut), 
                                    (task, gen_task),
                                    (task+'_UT', gen_ut_task_com),
                                    ('UT_UT', gen_ut_ut_com)]:
        cnt += 1
        default_config = deepcopy(DEFAULT_CONFIG)
        default_config['task'] = task
        default_config['fields'] = TRANSFORM_FIELDS[task]

        if cnt % 4 == 1:
            default_config['transform_methods'] = ['Prejudice']
        elif cnt % 4 == 2:
            default_config['transform_methods'] = gen_method(task)
        else:
            res = []
            for i in gen_method(task):
                if 'Prejudice' in i:
                    res.append(i)
            default_config['transform_methods'] = res

        for transformation in default_config['transform_methods']:
            transformation_list = [transformation] if not isinstance(transformation, list) else transformation
            for trans in transformation_list:
                if trans in TASK_CONFIGS and trans not in default_config['task_config']:
                    default_config['task_config'][trans] = TASK_CONFIGS[trans] if isinstance(TASK_CONFIGS[trans], list) else [TASK_CONFIGS[trans]]
            default_config['task_config']['LengthSubPopulation'] = TASK_CONFIGS['LengthSubPopulation']
            default_config['task_config']["PhraseSubPopulation"] = TASK_CONFIGS["PhraseSubPopulation"]
            default_config['task_config']["LMSubPopulation"] = TASK_CONFIGS["LMSubPopulation"]
            default_config['task_config']["PrejudiceSubPopulation"] = TASK_CONFIGS["PrejudiceSubPopulation"]
        out_json(default_config, "./{0}/{1}.json".format(task, config_type))


def out_json(json_obj, out_file):
    with open(out_file, "w+") as fo:
        json.dump(json_obj, fo, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    for task in TRANSFORMATIONS:
        if task != 'UT':
            gen_configs(task)
