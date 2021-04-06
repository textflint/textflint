r"""
Make short numbers grow into long numbers.
==========================================================
"""
__all__ = ["CnSwapNum"]
import random
from ..transformation import Transformation
from ....common.settings import NUM_LIST, NUM_FLAG1, NUM_FLAG2, \
    NUM_BEGIN, NUM_END


class CnSwapNum(Transformation):
    r"""
    Make short numbers grow into long numbers.

    Example::

        九百 -> 八百九十九

    """

    def __init__(self, **kwargs):
        r"""
        :param list num_list: the list which include all the number we need
            if you want to change it you must change NUM_LIST,
            NUM_FLAG1, NUM_FLAG2, NUM_BEGIN, NUM_END
        :param **kwargs:

        """
        super().__init__()
        self.num_list = NUM_LIST

    def __repr__(self):
        return 'CnSwapNum'

    def _transform(self, sample, n=5, **kwargs):
        r"""
        We randomly generated five sets of data.

        :param ~textflint.CWSSample sample: the data which need be changed
        :param int n: number of generated data
        :param **kwargs:
        :return: trans_sample a list of sample
        """
        # get sentence and label
        origin_sentence = sample.get_value('x')
        origin_label = sample.get_value('y')
        trans_sample = []

        for i in range(n):
            # change function
            change_pos, change_sentence, change_label = \
                self._get_transformations(origin_sentence, origin_label)
            if len(change_pos) == 0:
                return []
            change_sample = sample.replace_at_ranges(
                change_pos, change_sentence, change_label)
            trans_sample.append(change_sample)

        return trans_sample

    def _get_transformations(self, sentence, label):
        r"""
        Number change function.

        :param str sentence: chinese sentence
        :param list label: Chinese word segmentation tag
        :return list: three list include the pos which are changed the words
            which are changed and the labels which are changed

        """
        assert len(sentence) == len(label)

        start = 0
        change_pos = []
        change_sentence = []
        change_labels = []

        while start < len(sentence):
            # find the number
            if sentence[start] in self.num_list:
                if label[start] == 'S' and \
                        sentence[start - 1:start] != '第' \
                        and sentence[start] != '一':

                    if self.num_list.index(sentence[start]) < 10:
                        # if single number
                        # create a 至 b
                        ra = random.randint(1, 10)
                        rb = random.randint(1, 10)

                        while rb == ra:
                            rb = random.randint(1, 10)
                        if ra > rb:
                            tmp = ra
                            ra = rb
                            rb = tmp
                        change = self.num_list[ra] + '至' + self.num_list[rb]
                        change_label = ['B', 'M', 'E']
                        change_pos.append([start, start + 1])
                        change_sentence.append(change)
                        change_labels.append(change_label)
                    else:
                        # not a single number
                        change, change_label = self.number_change(
                            sentence, label, start, start)
                        if change != '':
                            change_pos.append([start, start + 1])
                            change_sentence.append(change)
                            change_labels.append(change_label)
                elif label[start] == 'B':
                    # Process numbers with length greater than 1
                    flag = 1
                    end = start
                    while label[end] != 'E':
                        end += 1
                        if sentence[end] not in self.num_list:
                            flag = 0
                            break
                    if flag:
                        change, change_label = self.number_change(
                            sentence, label, start, end)
                        if change:
                            change_pos.append([start, end + 1])
                            change_sentence.append(change)
                            change_labels.append(change_label)
                    start = end + 1
            start += 1

        return change_pos, change_sentence, change_labels

    def create_num(self, pos):
        r"""
        create chinese number

        :param int pos: the max length of number
        :return: the new number

        """
        # create chinese number
        if pos <= NUM_FLAG1:
            return str(self.num_list[random.randint(NUM_BEGIN, NUM_END)])

        res = ''
        if pos <= NUM_FLAG2:
            res += self.num_list[random.randint(NUM_BEGIN, NUM_END)] + \
                self.num_list[pos] + self.create_num(pos - 1)
            return res

        res += self.create_num(pos - 1) + \
            self.num_list[pos] + self.create_num(pos - 1)
        return res

    def number_change(self, sentence, label, start, end):
        r"""
        Digital conversion of start to end

        :param str sentence: the sentence to be changed
        :param list label: the tag of CWS
        :param int start: the start pos of the sentence
        :param int end: the end pos of the sentence
        :return str: the new number
        :return list: the label of new number

        """
        assert len(label) == len(sentence)
        # Digital conversion of start to end
        max_num = 0
        change = ''
        change_label = []

        for i in range(start, end + 1):
            max_num = max(self.num_list.index(sentence[i]), max_num)
        if end - start > 1 and max_num < 10:
            # Special numbers are not deformed
            return change, change_label
        change = self.create_num(max_num)
        seed = random.randint(0, 2)

        if len(change) == 1:
            seed = 0
        if seed == 1:
            change = change[:-1] + random.choice(['来', '余'])
        elif seed == 0:
            ca = self.create_num(max_num)
            cb = self.create_num(max_num)
            while ca == cb:
                cb = self.create_num(max_num)
            if self.compare(ca, cb):
                tmp = ca
                ca = cb
                cb = tmp
            change = ca + random.choice(['至', '到']) + cb
        if len(change) > 1:
            change_label = ['B'] + ['M'] * (len(change) - 2) + ['E']

        return change, change_label

    def compare(self, num1, num2):
        r"""
        compare two number

        :param str num1: the first number to be compared
        :param str num2: the second number to be compared
        """
        # compare two number
        for i in range(len(num1)):
            if num1[i] == num2[i]:
                continue
            return self.num_list.index(num1[i]) > self.num_list.index(num2[i])
        return True
