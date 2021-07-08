r"""
DeCLUTR sentence encoder class
=====================================
"""
import torch
from transformers import AutoModel, AutoTokenizer

from .validator import Validator

SEMANTIC_ENCODER = "johngiorgi/declutr-small"
__all__ = ['SentenceEncoding']


class SentenceEncoding(Validator):
    def __init__(
        self,
        origin_dataset,
        trans_dataset,
        fields
    ):
        r"""
        Constraint using similarity between sentence encodings of origin and
        translate where the text embeddings are created using the DeCLUTR.

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
        self.sim_metric = torch.nn.CosineSimilarity(dim=1)
        self.tokenizer = AutoTokenizer.from_pretrained(SEMANTIC_ENCODER)
        self.model = AutoModel.from_pretrained(SEMANTIC_ENCODER)

    def __repr__(self):
        return "SentenceEncoding"

    def validate(self, transformed_texts, reference_text):
        r"""
        Calculate the score

        :param str transformed_texts: transformed sentence
        :param str reference_text: origin sentence
        :return float: the score of two sentence

        """
        transformed_embeddings = self.encode(transformed_texts)
        reference_embeddings = self.encode(reference_text).expand(
            transformed_embeddings.size(0), -1)
        scores = self.sim_metric(transformed_embeddings, reference_embeddings)

        return float(scores.numpy())

    def encode(self, sentences):
        inputs = self.tokenizer(sentences, padding=True,
                                truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=False)
            sequence_output = outputs[0]

        embeddings = torch.sum(
            sequence_output * inputs["attention_mask"].unsqueeze(-1), dim=1
        ) / torch.clamp(torch.sum(inputs["attention_mask"],
                                  dim=1, keepdims=True), min=1e-9)

        return embeddings
