r"""
EntitySwap class for entity swap
"""
__all__ = ["SwapEnt"]
import random
from ....common.settings import LOWFREQ, MULTI_TYPE, TYPES
from ....common.utils.install import download_if_needed
from ....common.utils.load import json_loader
from ...transformation import Transformation
from ....input.component.sample import RESample


class SwapEnt(Transformation):
    r"""
    Replace entity mention with entity with same entity types

    """
    def __init__(
        self,
        type='lowfreq',
        **kwargs
    ):
        super().__init__()
        self.type = type
        if type == 'lowfreq':
            self.type_dict = json_loader(download_if_needed(LOWFREQ))
        elif type == 'multitype':
            self.type_dict = json_loader(download_if_needed(MULTI_TYPE))
        elif type == 'sametype':
            self.type_dict = json_loader(download_if_needed(TYPES))
        else:
            raise ValueError('illegal type name')

    def __repr__(self):
        return self.type

    def replace_en(self, types, index, token):
        r"""
        replace entity with random token span

        :param str types: entity type
        :param list index: entity index [start, end]
        :param list token: tokenized sentence
        :return Tuple(list, int): new sentence and \
        number of new entity words greater than old entity words
        """
        assert(isinstance(types, str)), \
            f"the type of 'type' should be string, got " \
            f"{type(types)} instead"
        assert(isinstance(index, list)), f"the type of 'index' " \
                                         f"should be list, got " \
                                         f"{type(index)} instead"
        assert(isinstance(token, list)), f"the type of 'token' " \
                                         f"should be list, got " \
                                         f"{type(token)} instead"
        assert(len(index)==2), f"the length of index " \
                               f"should be two, got length {len(index)} instead"
        assert(index[0]>=0 and index[1]<len(token)), \
            f"elements of index should not be negative or longer than " \
            f"the length of token, got input " \
            f"{index[0]}<0 or {index[1]}>= {len(token)}"
        length = 0
        if types in self.type_dict.keys():
            new_subj = random.choice(self.type_dict[types])
            token_before, token_after = token[:index[0]], token[index[1] + 1:]
            token = token_before + new_subj.split(" ") + token_after
            length = len(new_subj.split(" ")) - (index[1] - index[0] + 1)
        return token, length

    def subj_and_obj_transform(self, sample, n, entity):
        r"""
        transform both subject and object entities

        :param RESample sample: re_sample input
        :param int n: number of generated samples
        :return list: transformed sample list

        """
        assert(isinstance(n, int)), \
            f"the type of 'n' should be int, got {type(n)} instead"
        assert(isinstance(entity, list)), \
            f"the type of 'entity' should be list, got {type(entity)} instead"
        assert(isinstance(sample, RESample)), \
            f"the type of 'sample' should be RESample, got " \
            f"{type(sample)} instead"
        assert(len(entity) == 6), \
            f"the length of entity should be 6, got input length of " \
            f"{len(entity)} instead"
        assert(entity[0]<=entity[1] and entity[2]<=entity[3]), \
            f"start index of entity should not be greater than end index, got  \
                    {entity[0]}>{entity[1]} or {entity[2]}>{entity[3]} instead"
        assert(isinstance(entity[4], str) and isinstance(entity[5], str)), \
            f"last two elements of entity should be string, " \
            f"got type {entity[4]} and {entity[5]} instead"
        assert(entity[4] in self.type_dict.keys() and entity[5]
               in self.type_dict.keys()), \
            f"both entity types should be in the type dict, type " \
            f"{entity[4]} and {entity[5]} do not satisfy this requirement"

        trans_samples = []

        for i in range(n):
            sh, st, oh, ot, subj_type, obj_type = entity
            token, relation = sample.get_sent()
            trans_sample = {}

            if sh < oh:
                token, l1 = self.replace_en(subj_type, [sh, st], token)
                st, oh, ot = st + l1, oh + l1, ot + l1
                token, l2 = self.replace_en(obj_type, [oh, ot], token)
                ot = ot + l2
            elif oh < sh:
                token, l1 = self.replace_en(obj_type, [oh, ot], token)
                ot, sh, st = ot + l1, sh + l1, st + l1
                token, l2 = self.replace_en(subj_type, [sh, st], token)
                st = st + l2
            else:
                token, l = self.replace_en(subj_type, [sh, st], token)
                st, ot = st + l, ot + l
            trans_sample['x'] = token
            trans_sample['subj'], trans_sample['obj'], trans_sample['y'] = \
                [sh, st], [oh, ot], relation
            new_samples = sample.replace_sample_fields(trans_sample)
            trans_samples.append(new_samples)
        return trans_samples

    def single_transform(self, sample, n, entity):
        r"""
        transform  subject or object entity

        :param RESample sample: re_sample input
        :param int n: number of generated samples
        :return list: transformed sample list
        """
        assert(isinstance(n, int)), \
            f"the type of 'n' should be int, got {type(n)} instead"
        assert(isinstance(entity, list)), \
            f"the type of 'entity' should be list, got {type(entity)} instead"
        assert(isinstance(sample, RESample)), \
            f"the type of 'sample' should be RESample, got " \
            f"{type(sample)} instead"
        assert(len(entity) == 6), \
            f"the length of entity should be 6, got input length of " \
            f"{len(entity)} instead"
        assert(entity[0]<=entity[1] and entity[2]<=entity[3]), \
            f"start index of entity should not be greater than end index, got \
                    {entity[0]}>{entity[1]} or {entity[2]}>{entity[3]} instead"
        assert(isinstance(entity[4], str) and isinstance(entity[5], str)), \
            f"last two elements of entity should be string, got type " \
            f"{entity[4]} and {entity[5]} instead"
        assert(entity[4] not in self.type_dict.keys() or entity[5] not in
               self.type_dict.keys()), f"only one entity type should be " \
                                       f"in the type dict, " \
                                       f"{entity[4]} and {entity[5]} " \
                                       f"does not satisfy this requirement"
        trans_samples = []
        for i in range(n):
            sh, st, oh, ot, subj_type, obj_type = entity
            token, relation = sample.get_sent()
            trans_sample = {}
            if subj_type in self.type_dict.keys():
                token, l1 = self.replace_en(subj_type, [sh, st], token)
                st = st + l1
                if sh < oh:
                    oh, ot = oh + l1, ot + l1
                elif sh == oh:
                    ot = ot + l1
            else:
                token, l1 = self.replace_en(obj_type, [oh, ot], token)
                ot = ot + l1
                if sh > oh:
                    sh, st = sh + l1, st + l1
                elif sh == oh:
                    st = st + l1

            trans_sample['x'] = token
            trans_sample['subj'], trans_sample['obj'], trans_sample['y'] = \
                [sh, st], [oh, ot], relation
            new_samples = sample.replace_sample_fields(trans_sample)
            trans_samples.append(new_samples)
        return trans_samples

    def _transform(self, sample, n=1, **kwargs):
        r"""
        Transform text string according to its entities.

        :param RESample sample: re_sample input
        :param int n: number of generated samples
        :return list: transformed sample list

        """
        assert(isinstance(sample, RESample)), \
            f"the type of 'sample' should be RESample, " \
            f"got {type(sample)} instead"
        assert(isinstance(n, int)), \
            f"the type of 'n' should be int, got {type(n)} instead"
        sh, st, oh, ot = sample.get_en()
        subj_type, obj_type, _ = sample.get_type()
        if subj_type in self.type_dict.keys() and obj_type in \
                self.type_dict.keys():
            return self.subj_and_obj_transform(
                sample, n, [sh, st, oh, ot ,subj_type, obj_type])
        elif subj_type not in self.type_dict.keys() or \
                obj_type not in self.type_dict.keys():
            return self.single_transform(
                sample, n, [sh, st, oh, ot ,subj_type, obj_type])
        return [sample] * n
