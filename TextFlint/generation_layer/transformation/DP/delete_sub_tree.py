r"""
Remove a subtree in the sentence
============================================

"""

__all__ = ["DeleteSubTree"]

from ..transformation import Transformation


class DeleteSubTree(Transformation):
    r"""
    Transforms the input sentence by removing a subordinate clause.

    Example::

        original: "The bill intends to restrict the RTC to
            Treasury borrowings only, unless the agency receives
            specific congressional authorization."
        transformed: "The bill intends to restrict the RTC to
            Treasury borrowings only."

    """

    def __repr__(self):
        return 'DeleteSubtree'

    def _transform(self, sample, n=5, **kwargs):
        r"""
        Transform each sample case.

        :param ~DPSample sample:
        :return: transformed sample list.

        """
        subtrees = self.find_subtree(sample)
        if not subtrees:
            return []
        result = []

        for i, subtree in enumerate(subtrees):
            if i >= n:
                break
            else:
                sample_mod = sample.clone(sample)
                index = subtree[0] - 1
                for j in range(self.get_difference(subtree)):
                    sample_mod = sample_mod.delete_field_at_index('x', index)
                result.append(sample_mod)

        return result

    def find_subtree(self, sample):
        r"""
        Find all the subtrees that can be removed.

        :param ~DPSample sample:
        :return: A list of the subtrees, long to short.

        """
        words = sample.get_words('x')
        heads = sample.get_value('head')
        punc = []
        subtrees = []

        for i, word in enumerate(words):
            if word in (',', '.'):
                punc.append(i + 1)
        if len(punc) == 1:
            return None

        for i in range(len(punc) - 1):
            start = punc[i]
            end = punc[i + 1]
            flag = True
            bracket = 0
            interval = list(range(start + 1, end))

            for j, head in enumerate(heads):
                if int(head) in interval and (j + 1) not in interval:
                    flag = False
                    break

            for word in words[start:end - 1]:
                if word in ('-LRB-', '-RRB-'):
                    bracket += 1
            if flag is True and bracket % 2 == 0:
                subtrees.append((start, end))

        return sorted(subtrees, key=self.get_difference, reverse=True)

    @staticmethod
    def get_difference(num_pair):
        assert num_pair[0] < num_pair[1]
        return num_pair[1] - num_pair[0]

