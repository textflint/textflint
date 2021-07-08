r"""
Extract samples with high perplexity or low perplexity
============================================

"""
__all__ = ['LMSubPopulation']
import torch
import math
from ..subpopulation import SubPopulation


class LMSubPopulation(SubPopulation):
    r"""
    Filter samples based on text perplexity

    Example::

        sample 1: "I love textflint", score: 6.7
        sample 2: "I love TextFlinet", score: 6.34
    """
    def __init__(
            self,
            intervals=["0%", "20%"],
            device='cpu',
            max_sent_size=512
    ):
        if intervals is None:
            raise ValueError(
                'Intervals should be initialized for LMSubPopulation')
        super().__init__(intervals=intervals)
        self.tokenizer = None
        self.model = None
        self.device = device
        self.max_sent_size = max_sent_size

    def __repr__(self):
        return "LMSubPopulation-" + \
            str(self.intervals[0]) + "-" + str(self.intervals[1])

    def load_model(self):
        from transformers import GPT2Tokenizer, GPT2LMHeadModel
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained(
            'gpt2')
        self.model.to(self.device)

    def _score(self, sample, fields, **kwargs):
        r"""
        Calculate the score based on text perplexity

        :param sample: data sample
        :param list fields: list of field str
        :param kwargs:
        :return int: score for sample

        """
        if not self.model:
            self.load_model()

        perplexity = 0
        for field in fields:
            tokens = sample.get_words(field)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)[
                             :self.max_sent_size]
            tokens_tensor = torch.tensor(
                [indexed_tokens],
                dtype=torch.long,
                device=self.device)
            with torch.no_grad():
                output = self.model(tokens_tensor, labels=tokens_tensor)
            loss = output.loss
            perplexity += math.exp(loss.item())
        return perplexity
