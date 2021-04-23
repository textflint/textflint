import random
import copy

from .sample import Sample
from ..field import TextField
from ....common.settings import TASK_MASK


case_letters = [chr(i) for i in range(65, 91)]
uncase_letters = [chr(i) for i in range(97, 123)]
number = [str(i) for i in range(10)]

r"""
RESample class for sample formatting
"""
__all__ = ["RESample"]


class RESample(Sample):
    r"""
    transform and retrieve features of RESample

    """

    def __init__(
        self,
        data,
        origin=None,
        sample_id=None
    ):
        super().__init__(data, origin=origin, sample_id=sample_id)
        self.data = data

    def __repr__(self):
        return 'RESample'

    def check_data(self, data):
        r"""
        check whether type of data is correct

        :param dict data: data dict containing 'x', 'subj', 'obj' and 'y'

        """
        assert 'x' in data and isinstance(data['x'], list), \
            "x should be in data, and the type of x should be list"
        assert 'subj' in data and isinstance(data['subj'], list), \
            "subj should be in data, and the type of subj should be list"
        assert 'obj' in data and isinstance(data['obj'], list), \
            "obj should be in data, and the type of obj should be list"
        assert 'y' in data and isinstance(data['y'], str), \
            "y should be in data, and the type of y should be str"

    def is_legal(self):
        r"""
        Validate whether the sample is legal

        """
        sent_len = len(self.x.words)
        if not isinstance(self.y, str):
            return False
        if len(self.subj) != 2 or len(self.obj) != 2:
            return False
        if self.subj[0] < 0 or self.subj[1] >= sent_len or self.obj[0] < 0 \
                and self.obj[1] >= sent_len:
            return False
        if self.subj[0] > self.subj[1] or self.obj[0] > self.obj[1]:
            return False
        if self.subj[0] > self.subj[1] or self.obj[0] > self.obj[1]:
            return False
        for word in self.x.words:
            if word == '':
                return False
        return True

    def get_sent_ids(self):
        r"""
        Generate sentence ID

        :return: string: sentence ID

        """
        list = case_letters + uncase_letters + number
        num = random.sample(list, 10)
        str1 = ''
        value = str1.join(num)

        return value

    def load(self, data):
        r"""
        Convert data dict which contains essential information to SASample.

        :params: dict data: contains 'token', 'subj' ,'obj', 'relation' keys.

        """
        if type(data).__name__ == 'dict':
            if 'token' in data.keys():
                self.subj = [data['subj_start'], data['subj_end']]
                self.obj = [data['obj_start'], data['obj_end']]
                self.y = data['relation']
                data = data['token']
            else:
                self.subj = data['subj']
                self.obj = data['obj']
                self.y = data['y']
                self.data = data
                data = data['x']

        self.x = TextField(data)
        if not self.is_legal():
            raise ValueError("Data sample {0} is not legal, "
                             "entity or relation label is not in line."
                             .format(self.data))

        for i in range(self.subj[0], self.subj[1] + 1):
            self.x.set_mask(i, TASK_MASK)
        for i in range(self.obj[0], self.obj[1] + 1):
            self.x.set_mask(i, TASK_MASK)

    def get_dp(self):
        r"""
        get dependency parsing

        :return Tuple(list, list):  dependency tag of
            sentence and head of sentence

        """
        new_data = copy.deepcopy(self.data['x'])
        pars = TextField(new_data).dependency_parsing
        deprel, head = [], []
        for tuple in pars:
            deprel.append(tuple[-1])
            head.append(int(tuple[-2]))

        assert len(deprel) == len(self.data['x']), \
            'length of deprel should be the same as data'

        assert len(head) == len(self.data['x']), \
            'length of head should be the same as data'

        return deprel, head

    def get_en(self):
        r"""
        get entity index

        :return Tuple(int, int, int, int): start index of subject entity,
            end index of subject entity, start index of object entity
            and end index of object entity

        """
        self.sh, self.st = self.subj[0], self.subj[1]
        self.oh, self.ot = self.obj[0], self.obj[1]
        sh, st, oh, ot = copy.deepcopy(
            self.sh), copy.deepcopy(
            self.st), copy.deepcopy(
            self.oh), copy.deepcopy(
                self.ot)
        return sh, st, oh, ot

    def get_type(self):
        r"""
        get entity type

        :return Tuple(string, string): entity type of subject and
            entity type of object

        """
        self.ner = self.stan_ner_transform()
        if self.ner == 0:
            return 'O', 'O'
        self.subj_type = self.ner[self.subj[0]]
        self.obj_type = self.ner[self.obj[0]]

        return self.subj_type, self.obj_type, self.ner

    def get_sent(self):
        r"""
        get tokenized sentence

        :return Tuple(list, string): tokenized sentence and relation

        """
        return copy.deepcopy(self.data['x']), self.y

    def delete_field_at_indices(self, field, indices):
        r"""
        delete word of given indices in sentence

        :param string field: field to be operated on
        :param list indices: a list of index to be deleted

        :return dict: contains 'token', 'subj' ,'obj'  keys

        """
        sample = self.clone(self)
        text = self.data['x']

        for idx in indices:
            if type(idx).__name__ == 'int' and idx >= len(text) \
                    or type(idx).__name__ == 'list' and idx[-1] > len(text):
                print('index exceeds length')
                return sample

        sh, st = self.data['subj'][0], self.data['subj'][1]
        oh, ot = self.data['obj'][0], self.data['obj'][1]
        sub, ob, subj_to_delete, obj_to_delete, delete = [], [], [], [], []

        for idx in indices:
            if type(idx).__name__ == 'list':
                sub += range(idx[0], idx[1])
                ob += range(idx[0], idx[1])
                delete += range(idx[0], idx[1])
            else:
                sub.append(idx)
                ob.append(idx)
                delete.append(idx)
            subj_to_delete = [i for i in range(sh, st + 1) if i in sub]
            obj_to_delete = [i for i in range(oh, ot + 1) if i in ob]
        new_text, new_data = [], {}
        delete = list(set(delete))

        if len(subj_to_delete) > 0 or len(obj_to_delete) > 0:
            print('You should not delete entity')
            return sample
        subj_before, obj_before = 0, 0

        for idx in delete:
            if idx < sh:
                subj_before += 1
            if idx < oh:
                obj_before += 1
        for idx in range(len(text)):
            if idx not in delete:
                new_text.append(text[idx])
        new_data['x'], new_data['subj'], new_data['obj'], new_data['y'] = \
            new_text, [sh - subj_before, st - subj_before], \
            [oh - obj_before, ot - obj_before], self.data['y']
        sample.load(new_data)

        return sample

    def insert_field_after_indices(self, field, indices, new_item):
        r"""
        insert word before given indices in sentence

        :param string field: field to be operated on
        :param list indices: a list of index to be inserted
        :param list new_item: list of items to be inserted

        :return dict: contains 'token', 'subj' ,'obj'  keys

        """
        for i, item in enumerate(new_item):
            if type(item).__name__ == 'list':
                new_item[i] = ' '.join(item)
        sample = self.clone(self)
        text = self.data['x']
        for idx in indices:
            if idx >= len(text):
                print('index exceeds length')
                return sample

        sh, st = self.data['subj'][0], self.data['subj'][1]
        oh, ot = self.data['obj'][0], self.data['obj'][1]
        new_text, new_data = [], {}

        subj_before, obj_before = 0, 0
        for i, idx in enumerate(indices):
            if idx >= sh and idx <= st and st > sh:
                print('you should not change entity')
                return sample
            if idx >= oh and idx <= ot and ot > oh:
                print('you should not change entity')
                return sample
            if idx < sh:
                l = len(new_item[i].split(" "))
                subj_before += l
            if idx < oh:
                l = len(new_item[i].split(" "))
                obj_before += l

        for i in range(len(text)):
            new_text.append(text[i])
            if i in indices:
                idx = indices.index(i)
                to_insert = new_item[idx].split(" ")
                new_text += to_insert

        new_data['x'], new_data['subj'], new_data['obj'], new_data['y'] = \
            new_text, [sh + subj_before, st + subj_before], \
            [oh + obj_before, ot + obj_before], self.data['y']
        sample.load(new_data)
        return sample

    def insert_field_before_indices(self, field, indices, new_item):
        r"""
        insert word after given indices in sentence

        :param string field: field to be operated on
        :param list indices: a list of index to be inserted
        :param list new_item: list of items to be inserted

        :return dict: contains 'token', 'subj' ,'obj'  keys

        """
        sample = self.clone(self)
        text = self.data['x']

        for i, item in enumerate(new_item):
            if type(item).__name__ == 'list':
                new_item[i] = ' '.join(item)
        for idx in indices:
            if idx >= len(text):
                print('index exceeds length')
                return sample
        sh, st = self.data['subj'][0], self.data['subj'][1]
        oh, ot = self.data['obj'][0], self.data['obj'][1]
        new_text, new_data = [], {}
        subj_before, obj_before = 0, 0
        for i, idx in enumerate(indices):
            if idx >= sh and idx <= st and st > sh:
                print('you should not change entity')
                return sample
            if idx >= oh and idx <= ot and ot > oh:
                print('you should not change entity')
                return sample
            if idx < sh:
                l = len(new_item[i].split(" "))
                subj_before += l
            if idx < oh:
                l = len(new_item[i].split(" "))
                obj_before += l

        for i in range(len(text)):
            if i in indices:
                idx = indices.index(i)
                to_insert = new_item[idx].split(" ")
                new_text += to_insert
            new_text.append(text[i])

        new_data['x'], new_data['subj'], new_data['obj'], new_data['y'] = \
            new_text, [sh + subj_before, st + subj_before], \
            [oh + obj_before, ot + obj_before], self.data['y']
        sample.load(new_data)

        return sample

    def replace_sample_fields(self, data):
        r"""
        replace sample fields for RE transformation

        :param dict data: contains transformed x, subj, obj keys
        :return RESample: transformed sample

        """
        sample = self.clone(self)
        sample.load(data)

        return sample

    def stan_ner_transform(self):
        r"""
        Generate ner list

        :return list: ner tags

        """
        text = self.data['x']
        ners = ['O'] * len(text)
        ner = self.x.ner
        for en in ner:
            start = en[1]
            end = en[2]
            type = en[3]
            for i in range(start, end):
                ners[i] = type
        return ners

    def dump(self):
        r"""
        output data sample

        :return dict: containing x, subj, obj, y and sample_id

        """
        if not self.is_legal():
            raise ValueError("Data sample {0} is not legal, "
                             "entity index exceeds sentence length."
                             .format(self.data))

        return {
            'x': self.x.words,
            'subj': self.subj,
            'obj': self.obj,
            'y': self.y,
            'sample_id': self.sample_id
        }
