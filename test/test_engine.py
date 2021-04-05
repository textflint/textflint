import unittest
import shutil
import os

from TextFlint.engine import Engine
from TextFlint.common.utils.install import download_if_needed
from TextFlint.input_layer.config.config import Config
from TextFlint.common.settings import UNMATCH_UT_TRANSFORMATIONS, \
    TASK_TRANSFORMATIONS
tasks = ['ABSA', 'COREF', 'CWS', 'DP',
                 'MRC', 'NER', 'NLI', 'POS', 'RE', 'SA', 'SM']
test_config = {
    "max_trans": 1,
    "return_unk": True,
    "transformation_config": {},
    "transformation_methods": [],
    "subpopulation_methods": []
    }


def get_test(task):
    test_config['transformation_methods'] = TASK_TRANSFORMATIONS[task]
    # for ut in TASK_TRANSFORMATIONS['UT']:
    #     if ut not in UNMATCH_UT_TRANSFORMATIONS['SA']:
    #         test_config['transformation_methods'].append(ut)
    test_config['task'] = task
    config = Config.from_dict(test_config)
    out_dir_path = os.path.normcase('./test_result_test/' + task + '/')
    engine = Engine(task)
    engine.run(os.path.normcase(download_if_needed('TEST/' + task + '.json')),
               out_dir_path, config)
    shutil.rmtree(out_dir_path)


class TestEngine(unittest.TestCase):
    def test_absa_engine(self):
        get_test('ABSA')

    def test_sa_engine(self):
        get_test('SA')

    def test_cws_engin(self):
        get_test('CWS')

    def test_mrc_engine(self):
        get_test('MRC')

    def test_nli_engine(self):
        get_test('NLI')

    def test_sm_engine(self):
        get_test('SM')

    def test_pos_engine(self):
        get_test('POS')

    def test_re_engine(self):
        get_test('RE')

    def test_ner_engine(self):
        get_test('NER')

    def test_dp_engine(self):
        get_test('DP')

    def test_coref_engine(self):
        get_test('COREF')


if __name__ == "__main__":
    unittest.main()
    os.remove(os.path.normcase('./test_result_test'))
