r"""
GPT-2 language model perplexity class
=====================================
"""
import math
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from .validator import Validator
__all__ = ['GPT2Perplexity']


class GPT2Perplexity(Validator):
    def __init__(
        self,
        origin_dataset,
        trans_dataset,
        fields
    ):
        r"""
        Constraint using OpenAI GPT2 language model perplexity of x_adv.

        :param ~textflint.input.dataset origin_dataset:
                the dataset of origin sample
        :param ~textflint.input.dataset trans_dataset:
            the dataset of translate sample
        :param str|list fields: the name of the origin field need compare.

        """
        super().__init__(
            origin_dataset,
            trans_dataset,
            fields
        )
        r"""
        :param ~textflint.input.dataset origin_dataset:
                the dataset of origin sample
        :param ~textflint.input.dataset trans_dataset:
            the dataset of translate sample
        :param str|list fields: the name of the origin field need compare.
        
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def __repr__(self):
        return "GPT2Perplexity"

    def validate(self, transformed_text, reference_text):
        r"""
        Calculate the score

        :param str transformed_text: transformed sentence
        :param str reference_text: origin sentence
        :return float: the score of two sentence
        """
        return min(self.perplexity(reference_text) /
                   self.perplexity(transformed_text), 1)

    def perplexity(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt")
        if len(inputs["input_ids"][0] > 1024):
            new_ids = inputs["input_ids"][0][:1024]
            new_mask = inputs['attention_mask'][0][:1024]
            inputs = {"input_ids": torch.stack([new_ids]),
                      'attention_mask': torch.stack([new_mask])}

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"],
                                 return_dict=True)
        return math.exp(outputs.loss.item())
