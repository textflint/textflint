from nltk import Tree
from ..subpopulation import SubPopulation


# TODO
class SubtreeSubPopulation(SubPopulation):
    def __init__(self):
        # TODO
        super().__init__()

    def __repr__(self):
        return "SubtreeSubPopulation"

    def score(self, sample, fields, **kwargs):
        import fuzzywuzzy.fuzz as fuzz

        """Calculate the score based on constituency subtree match between texts
        Args:
            sample: data sample
            fields: list of str
            **kwargs:

        Returns: int, score for sample

        Returns:

        """
        assert len(fields) == 2, "Need two fields for this subpopulation, given {0}".format(fields)
        text_0 = sample.get_text(fields[0])
        text_1 = sample.get_text(fields[1])
        # TODO,
        tree_0 = Tree.fromstring(self.text_processor.get_parser(text_0).replace('\n', ''))
        subtrees = set([str(t).replace('\n', '').replace(' ', '').lower() for t in tree_0.subtrees()])
        tree_1 = self.text_processor.get_parser(text_1).replace('\n', '').replace(' ', '') \
            .replace("(..)", "").replace("(,,)", "").lower()

        return max([fuzz.partial_ratio(tree_1, subtree) for subtree in subtrees])
