from ..subpopulation import SubPopulation


# TODO
class ConstituencySubPopulation(SubPopulation):

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "ConstituencySubPopulation"

    def score(self, sample, fields, **kwargs):
        """Calculate the score based on the constituency tree match between texts
        Args:
            sample: data sample
            fields: list of str
            **kwargs:

        Returns: int, score for sample

        """
        import fuzzywuzzy.fuzz as fuzz
        assert len(fields) == 2, "Need two fields for this subpopualtion, " \
                                 "given {0}".format(fields)
        text_0 = sample.get_text(fields[0])
        text_1 = sample.get_text(fields[1])
        tree_0 = self.text_processor.get_parser(text_0).replace('\n', '')\
            .replace('(', '').replace(')', '').replace(' ', '')
        tree_1 = self.text_processor.get_parser(text_1).replace('\n', '')\
            .replace('(', '').replace(')', '').replace(' ', '')

        return fuzz.partial_token_set_ratio(tree_0, tree_1)
