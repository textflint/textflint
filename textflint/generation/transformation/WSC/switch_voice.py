__all__ = ["SwitchVoice"]
from nltk.wsd import lesk

from ...transformation import Transformation
from ....common.utils.install import download_if_needed
from ....common.utils.load import load_jsonlines
from ....common.utils import logger
from textflint.input.component.sample.wsc_sample import WSCSample
from ....common.settings import FILE_NAME_DICT
import json

class SwitchVoice(Transformation):
    def __init__(self, wsc_task='SwitchVoice', **kwargs):
        r"""
        :param string swap_type: the swap type in
        ['CrossCategory', 'OOV', 'SwapLonger']

        :param string res_path: dir for vocab/dict
        """

        super().__init__()
        trans_path = FILE_NAME_DICT[wsc_task + '_PATH']
        res_dic = load_jsonlines(download_if_needed(trans_path))
        self.sample_dict = {}
        for data in res_dic:
            data = json.loads(data)
            sample = WSCSample(data)
            index = data['index']
            self.sample_dict[index] = sample

    def __repr__(self):
        return 'SwitchVoice'


    def _transform(self, sample, n=1, **kwargs):
        wsc_samples = []
        # find the sample in trans_file by search "index"
        idx = sample.index
        if idx in self.sample_dict:
            trans_sample = self.sample_dict[idx]
            trans_sample = WSCSample.clone(trans_sample)
            trans_sample.origin = False
            wsc_samples.append(trans_sample)
        else:
            logger.info("The data can not be transformed by SwitchVoice")

        return wsc_samples
