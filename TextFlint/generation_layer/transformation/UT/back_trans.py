r"""
Back translation class
==========================================================
"""

__all__ = ['BackTrans']

import torch

from ..transformation import Transformation
from ....common import device as default_device
from ....common.settings import TRANS_FROM_MODEL, TRANS_TO_MODEL


class BackTrans(Transformation):
    r"""
    Back Translation with hugging-face translation models.
    A sentence can only be transformed into one sentence at most.

    """
    def __init__(
        self,
        from_model_name=None,
        to_model_name=None,
        device=None,
        **kwargs
    ):
        r"""
        :param str from_model_name: model to translate original language to
            target language
        :param str to_model_name: model to translate target language to
            original language
        :param device: indicate utilize cpu or which gpu device to
            run neural network

        """
        super().__init__()
        self.device = self.get_device(device) if device else default_device
        self.from_model_name = from_model_name if from_model_name else \
            TRANS_FROM_MODEL
        self.to_model_name = to_model_name if to_model_name else TRANS_TO_MODEL
        self.from_model = None
        self.from_tokenizer = None
        self.to_model = None
        self.to_tokenizer = None

    def __repr__(self):
        return 'BackTrans'

    @staticmethod
    def get_device(device):
        r"""
        Get gpu or cpu device.

        :param str device: device string
                           "cpu" means use cpu device.
                           "cuda:0" means use gpu device which index is 0.
        :return: device in torch.

        """
        if "cuda" not in device:
            return torch.device("cpu")
        else:
            return torch.device(device)

    def get_model(self):
        # accelerate load efficiency
        from transformers import FSMTForConditionalGeneration, FSMTTokenizer
        generation = FSMTForConditionalGeneration
        tokenizer = FSMTTokenizer

        """ Load models of translation. """
        self.from_tokenizer = tokenizer.from_pretrained(self.from_model_name)
        self.from_model = generation.from_pretrained(self.from_model_name)
        self.to_tokenizer = tokenizer.from_pretrained(self.to_model_name)
        self.to_model = generation.from_pretrained(self.to_model_name)
        self.from_model.to(self.device)
        self.to_model.to(self.device)

    def _transform(self, sample, n=1, field='x', **kwargs):
        if self.to_model is None:
            self.get_model()
        text = sample.get_text(field)
        # translate
        input_ids = self.from_tokenizer.encode(text, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        outputs = self.from_model.generate(input_ids)
        translated_text = self.from_tokenizer.decode(
            outputs[0], skip_special_tokens=True)
        # back_translate
        input_ids = self.to_tokenizer.encode(
            translated_text, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        outputs = self.to_model.generate(input_ids)
        back_translated_text = self.to_tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        return [sample.replace_field(field, back_translated_text)]

