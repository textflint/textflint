from ..subpopulation import SubPopulation


# TODO
class LexicalSubPopulation(SubPopulation):

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "LexicalSubPopulation"

    def score(self, sample, fields, **kwargs):
        """Calculate the score based on the lexical overlap of hypothesis and premise
        Args:
            sample: data sample
            fields: list of str
            **kwargs:

        Returns: int, score for sample

        """
        assert len(fields) == 2, "Need two field for this subpopulation, given {0}".format(fields)
        tokens_0 = set([tok.lower() for tok in sample.get_words(fields[0])])
        tokens_1 = set([tok.lower() for tok in sample.get_words(fields[1])])

        return len(tokens_0.intersection(tokens_1)) / float(len(tokens_0.union(tokens_1)))
