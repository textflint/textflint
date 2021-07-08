r"""
Coref Sample Class
============================================
"""

from ....common.settings import ORIGIN, TASK_MASK
from ....common.utils.fp_utils import *
from ....common.utils.shift_utils import *
from .sample import Sample
from ..field import ListField, TextField

__all__ = ['CorefSample']


class CorefSample(Sample):
    r"""
    Coref Sample
    """
    def __init__(self, data, origin=None, sample_id=None):
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'CorefSample'

    # basics

    def check_data(self, data):
        r"""
        Check if `data` is a conll-dict and is ready to be predicted.

        :param None|dict data:
            Must have key: sentences, clusters
            May have key: doc_key, speakers, constituents, ner
        :return:

        """
        if data == None:
            return
        assert isinstance(data, dict), "To be loaded by CorefSample: not a dict"
        # doc_key: string
        if "doc_key" in data:
            assert isinstance(data["doc_key"], str), \
                "To be loaded by CorefSample: `doc_key` is not a str"
        # sentences: 2nd list of str; word list list
        assert "sentences" in data and isinstance(data["sentences"], list), \
            "To be loaded by CorefSample: `sentences` is not a list"

        if len(data["sentences"]) > 0:
            assert isinstance(data["sentences"][0], list), \
                "To be loaded by CorefSample: " \
                "`sentences` is not a 2nd list"
            assert isinstance(data["sentences"][0][0], str), \
                "To be loaded by CorefSample: " \
                "`sentences` is not a word list list"

        # speakers: 2nd list of str; word list list
        if "speakers" in data:
            assert isinstance(data["speakers"], list), \
                "To be loaded by CorefSample: " \
                "`speakers` is not a list"
            if len(data["speakers"]) > 0:
                assert isinstance(data["speakers"][0], list), \
                    "To be loaded by CorefSample: " \
                    "`speakers` is not a 2nd list"
                assert isinstance(data["speakers"][0][0], str), \
                    "To be loaded by CorefSample: " \
                    "`speakers` is not a word list list"
        # clusters: 2nd list of span([int, int]); cluster list
        assert "clusters" in data and isinstance(data["clusters"], list), \
            "To be loaded by CorefSample: `clusters` is not a list"

        if len(data["clusters"]) > 0:
            for cluster in data["clusters"]:
                assert isinstance(cluster, list), \
                    "To be loaded by CorefSample: " \
                    "cluster in `clusters` is not a list"
                assert len(cluster) > 1, \
                    "To be loaded by CorefSample: " \
                    "cluster in `clusters` has < 2 spans"
                assert isinstance(cluster[0][0], int), \
                    "To be loaded by CorefSample: " \
                    "cluster in `clusters` is not a span list"

        # constituents: list of tag([int, int, str])
        if "constituents" in data:
            assert isinstance(data["constituents"], list), \
                "To be loaded by CorefSample: " \
                "`constituents` is not a list"
            if len(data["constituents"]) > 0:
                assert isinstance(data["constituents"][0], list), \
                    "To be loaded by CorefSample: " \
                    "constituent in `constituents` is not a list"
                assert isinstance(data["constituents"][0][0], int), \
                    "To be loaded by CorefSample: " \
                    "constituent in `constituents` is not a [b, e, label]"
                assert isinstance(data["constituents"][0][2], str), \
                    "To be loaded by CorefSample: " \
                    "constituent in `constituents` is not a [b, e, label]"

        # ner: list of tag([int, int, str])
        if "ner" in data:
            assert isinstance(data["ner"], list), \
                "To be loaded by CorefSample: `ner` is not a list"
            if len(data["ner"]) > 0:
                assert isinstance(data["ner"][0], list), \
                    "To be loaded by CorefSample: " \
                    "constituent in `ner` is not a list"
                assert isinstance(data["ner"][0][0], int), \
                    "To be loaded by CorefSample: " \
                    "constituent in `ner` is not a [b, e, label]"
                assert isinstance(data["ner"][0][2], str), \
                    "To be loaded by CorefSample: " \
                    "constituent in `ner` is not a [b, e, label]"

    def is_legal(self):
        r"""
        Validate whether the sample is legal.

        """
        data = self.dump(with_check=False)
        try:
            self.check_data(data)
        except Exception as e:
            return False
        return True

    def load(self, data):
        r"""
        Convert a conll-dict to CorefSample.

        :param None|dict data: None, or a conll-style dict
            Must have key: sentences, clusters
            May have key: doc_key, speakers, constituents, ner
        :return:

        """
        if data == None:
            # raise ValueError("In coref_sample: load_data, data == None")
            self.doc_key = ""
            self.x = TextField([])
            self.sen_map = []
            self.doc_sp = TextField([])
            self.clusters = ListField([])
            self.constituents = ListField([])
            self.ner = ListField([])
            return
        self.check_data(data)
        # doc_key
        self.doc_key = data["doc_key"] if "doc_key" in data else ""
        # sample mask, x, doc_sp
        x, sen_map = self.sens2doc(data["sentences"])
        mask = [ORIGIN] * len(x)

        for cluster in data["clusters"]:
            for [b, e] in cluster:
                for i in range(b, e+1):
                    mask[i] = TASK_MASK
        self.x = TextField(x, mask=mask)
        self.sen_map = sen_map

        if "speakers" in data:
            doc_sp, sen_map = self.sens2doc(data["speakers"])
        else:
            doc_sp = ["sp"] * len(x)
        self.doc_sp = TextField(doc_sp, mask=mask)
        # clusters
        self.clusters = ListField(data["clusters"])
        # constituents, ner
        constituents = data["constituents"] if "constituents" in data else []
        ner = data["ner"] if "ner" in data else []
        self.constituents = ListField(constituents)
        self.ner = ListField(ner)

    def dump(self, with_check=True):
        r"""
        Dump a CorefSample to a conll-dict.

        :param bool with_check: whether the dumped conll-dict should be checked
        :return dict ret_dict: a conll-style dict

        """
        ret_dict = dict()
        ret_dict["doc_key"] = self.doc_key
        ret_dict["sentences"] = self.doc2sens(
            self.x.words, self.sen_map)
        ret_dict["speakers"] = self.doc2sens(
            self.doc_sp.words, self.sen_map)
        ret_dict["clusters"] = self.clusters.field_value
        ret_dict["constituents"] = self.constituents.field_value
        ret_dict["ner"] = self.ner.field_value
        # append for identify sample
        ret_dict["sample_id"] = self.sample_id
        if with_check:
            assert len(self.x.words) == len(self.x.mask)
            assert len(self.doc_sp.words) == len(self.doc_sp.mask)
            assert len(self.x.words) == len(self.doc_sp.words)
            self.check_data(ret_dict)
        return ret_dict

    # debug samples and methods

    def pretty_print(self, show="Sample:"):
        r"""
        A pretty-printer for CorefSample. Print useful sample information
        by calling this function.

        :param str show: optional, the welcome information of
            printing this sample

        """
        print(show)
        doc = [word if mask == 0 else word+"_" +
               str(mask) for word, mask in zip(self.x.words, self.x.mask)]
        print("Sentences:")
        for s in self.doc2sens(doc, self.sen_map):
            print(s)
        clusters = self.clusters.field_value
        clusters = [
            [
                ((b, doc[b]), (e, doc[e]))
                for [b, e] in cluster]
            for cluster in clusters]
        print("Clusters:")
        print(clusters)

    # basic methods

    def num_sentences(self):
        r"""
        the number of sentences in this sample

        :param:
        :return int: the number of sentences in this sample

        """
        return len(self.sen_map)

    def get_kth_sen(self, k):
        r""" get the kth sen as a word list

        :param int k: sen id
        :return list: kth sen, word list

        """
        ret_part = self.part_conll([k])
        return ret_part.x.words

    def eqlen_sen_map(self):
        r"""
        Generate [0, 0, 1, 1, 1, 2, 2]
        from self.sen_map = [2, 3, 2]

        :param:
        :return list: sentence mapping with equal length to x,
            like [0, 0, 1, 1, 1, 2, 2]

        """
        eqlen_sen_map = []
        for i in range(len(self.sen_map)):
            eqlen_sen_map.extend([i] * self.sen_map[i])
        return eqlen_sen_map

    def index_in_sen(self, idx):
        r"""
        For the given word idx, determine which sen it is in.

        :param int idx: word idx
        :return int: sen_idx, which sentence is word idx in

        """
        return self.eqlen_sen_map()[idx]

    @staticmethod
    def sens2doc(sens):
        r"""
        Given an 2nd list of str (word list list),
        concat it and records the length of each sentence

        :param list sens: 2nd list of str (word list list)
        :returns (list, list): x as list of str (word list),
            sen_map as list of int (sen len list)

        """
        x = concat(sens)
        sen_map = [len(sen) for sen in sens]
        return x, sen_map

    @staticmethod
    def doc2sens(x, sen_map):
        """
        Given x and sen_map, return sens.
        Inverse to `sens2doc`.

        :param list x: list of str (word list)
        :param list sen_map: list of int (sen len list)
        :return list: sens as 2nd list of str (word list list)

        """
        curr_idx = 0
        sens = []
        for i in range(len(sen_map)):
            sen_len = sen_map[i]
            if sen_len == 0: continue
            sen = x[curr_idx: curr_idx+sen_len]
            sens.append(sen)
            curr_idx += sen_len
        return sens

    # methods for word-level modification: insert, delete, replace
    def insert_field_before_indices(self, field, indices, items):
        r"""
        Insert items of given scopes before indices of field value simutaneously

        :param str field: transformed field
        :param list indices: indices of insert positions
        :param list items: insert items
        :return ~textflint.CorefSample: modified sample

        """
        # arg type check
        assert field == 'x'
        for (idx, item) in zip(indices, items):
            assert isinstance(idx, int)
            assert isinstance(item, (str, list))
            if isinstance(item, list) and len(item) > 0:
                assert isinstance(item[0], str)
        # start
        sample = self.clone(self)
        # x
        field_obj = getattr(sample, 'x')
        assert isinstance(field_obj, TextField)
        rep_obj = field_obj.insert_before_indices(indices, items)
        setattr(sample, 'x', rep_obj)

        # doc_sp
        items_sp = recur_ap(lambda x: "sp_ins", items)
        field_obj = getattr(sample, 'doc_sp')
        assert isinstance(field_obj, TextField)
        rep_obj = field_obj.insert_before_indices(indices, items_sp)
        setattr(sample, 'doc_sp', rep_obj)
        # calc for item lengths (item_shifts)
        item_shifts = [1 if isinstance(
            item, str) else len(item) for item in items]

        # calc for shift
        shifts = [shift_maker(idx, item_shift)
                  for (idx, item_shift) in zip(indices, item_shifts)]
        index_shift = shift_decor(shift_collector(shifts))

        # clusters, constituents, ner
        clusters = getattr(sample, 'clusters').field_value
        constituents = getattr(sample, 'constituents').field_value
        ner = getattr(sample, 'ner').field_value
        setattr(sample, 'clusters', ListField(recur_ap(index_shift, clusters)))
        setattr(sample, 'constituents', ListField(
            recur_ap(index_shift, constituents)))
        setattr(sample, 'ner', ListField(recur_ap(index_shift, ner)))

        # sen_map
        sen_map = getattr(sample, 'sen_map')
        for (idx, item_shift) in zip(indices, item_shifts):
            sen_idx = self.index_in_sen(idx)
            sen_map[sen_idx] += item_shift
        setattr(sample, 'sen_map', sen_map)

        return sample

    def insert_field_after_indices(self, field, indices, items):
        r"""
        Insert items of given scopes after indices of field value simutaneously.

        :param str field: transformed field
        :param list indices: indices of insert positions
        :param list items: insert items
        :return ~textflint.CorefSample: modified sample

        """
        # arg type check
        assert field == 'x'
        for (idx, item) in zip(indices, items):
            assert isinstance(idx, int)
            assert isinstance(item, (str, list))
            if isinstance(item, list) and len(item) > 0:
                assert isinstance(item[0], str)
        # start
        sample = self.clone(self)
        # x
        field_obj = getattr(sample, 'x')
        assert isinstance(field_obj, TextField)
        rep_obj = field_obj.insert_after_indices(indices, items)
        setattr(sample, 'x', rep_obj)

        # doc_sp
        items_sp = recur_ap(lambda x: "sp_ins", items)
        field_obj = getattr(sample, 'doc_sp')
        assert isinstance(field_obj, TextField)
        rep_obj = field_obj.insert_after_indices(indices, items_sp)
        setattr(sample, 'doc_sp', rep_obj)

        # calc for item lengths (item_shifts)
        item_shifts = [1 if isinstance(
            item, str) else len(item) for item in items]
        # calc for shift
        shifts = [shift_maker(idx+1, item_shift)
                  for (idx, item_shift) in zip(indices, item_shifts)]
        index_shift = shift_decor(shift_collector(shifts))

        # clusters, constituents, ner
        clusters = getattr(sample, 'clusters').field_value
        constituents = getattr(sample, 'constituents').field_value
        ner = getattr(sample, 'ner').field_value
        setattr(sample, 'clusters', ListField(recur_ap(index_shift, clusters)))
        setattr(sample, 'constituents', ListField(
            recur_ap(index_shift, constituents)))
        setattr(sample, 'ner', ListField(recur_ap(index_shift, ner)))

        # sen_map
        sen_map = getattr(sample, 'sen_map')
        for (idx, item_shift) in zip(indices, item_shifts):
            sen_idx = self.index_in_sen(idx)
            sen_map[sen_idx] += item_shift
        setattr(sample, 'sen_map', sen_map)

        return sample

    def delete_field_at_indices(self, field, indices):
        r""" Delete items of given scopes of field value.

        :param str field: transformed field
        :param list indices: indices of delete positions
        :return ~textflint.CorefSample: modified sample

        """
        # arg type check
        assert field == 'x'
        for span in indices:
            assert isinstance(span, (int, list, slice))
            if isinstance(span, list):
                assert len(span) == 2
            if isinstance(span, slice):
                assert span.start >= 0
                assert span.stop >= span.start
                assert span.step == 1 or span.step == None

        # arg convert
        indices_tmp = []
        for span in indices:
            if isinstance(span, int):
                indices_tmp.append(span)
            elif isinstance(span, list):
                indices_tmp.extend(range(span[0], span[1]))
            elif isinstance(span, slice):
                indices_tmp.extend(range(span.stop)[span])

        indices = sorted(list(set(indices_tmp)))
        # start
        sample = self.clone(self)
        # x
        field_obj = getattr(sample, 'x')
        assert isinstance(field_obj, TextField)
        rep_obj = field_obj.delete_at_indices(indices)
        setattr(sample, 'x', rep_obj)
        # doc_sp
        field_obj = getattr(sample, 'doc_sp')
        assert isinstance(field_obj, TextField)
        rep_obj = field_obj.delete_at_indices(indices)
        setattr(sample, 'doc_sp', rep_obj)
        # calc for item lengths (item_shifts)
        item_shifts = [-1] * len(indices)
        # calc for shift
        shifts = [shift_maker(idx, item_shift)
                  for (idx, item_shift) in zip(indices, item_shifts)]
        index_shift = shift_decor(shift_collector(shifts))
        # clusters, constituents, ner

        def preserve(span):
            return not span[0] in indices

        clusters = [[span for span in cluster if preserve(
            span)] for cluster in getattr(sample, 'clusters').field_value]
        constituents = [span for span in getattr(
            sample, 'constituents').field_value if preserve(span)]
        ner = [span for span in getattr(
            sample, 'ner').field_value if preserve(span)]

        setattr(sample, 'clusters', ListField(recur_ap(index_shift, clusters)))
        setattr(sample, 'constituents', ListField(
            recur_ap(index_shift, constituents)))
        setattr(sample, 'ner', ListField(recur_ap(index_shift, ner)))

        # sen_map
        sen_map = getattr(sample, 'sen_map')
        for (idx, item_shift) in zip(indices, item_shifts):
            sen_idx = self.index_in_sen(idx)
            sen_map[sen_idx] += item_shift
        setattr(sample, 'sen_map', sen_map)

        # sample + remove invalid cluster labels
        return CorefPartSample(
            sample.dump(with_check=False)).remove_invalid_corefs_from_part()

    def replace_field_at_indices(self, field, indices, items):
        r"""

        Replace scope items of field value with items.
        :param str field: transformed field
        :param list indices: indices of delete positions
        :param list items: insert items
        :return ~textflint.CorefSample: modified sample

        """
        # arg type check
        assert field == 'x'
        for (span, item) in zip(indices, items):
            if isinstance(span, int):
                len_span = 1
            elif isinstance(span, list):
                assert len(span) == 2
                len_span = span[1] - span[0]
            elif isinstance(span, slice):
                assert span.start >= 0
                assert span.stop >= span.start
                assert span.step == 1 or span.step == None
                len_span = span.stop - span.start
            else:
                raise AssertionError

            if isinstance(item, str):
                len_item = 1
            elif isinstance(item, list):
                if len(item) > 0:
                    assert isinstance(item[0], str)
                len_item = len(item)
            else:
                raise AssertionError
            assert len_span == len_item

        # start
        sample = self.clone(self)
        # x
        field_obj = getattr(sample, 'x')
        assert isinstance(field_obj, TextField)
        rep_obj = field_obj.replace_at_indices(indices, items)
        setattr(sample, 'x', rep_obj)

        # doc_sp
        items_sp = recur_ap(lambda x: "sp_repl", items)
        field_obj = getattr(sample, 'doc_sp')
        assert isinstance(field_obj, TextField)
        rep_obj = field_obj.replace_at_indices(indices, items_sp)
        setattr(sample, 'doc_sp', rep_obj)

        return CorefPartSample(
            sample.dump(with_check=False)).remove_invalid_corefs_from_part()

    # methods for making coref sample
    @staticmethod
    def concat_conlls(*args):
        r"""
        Given several CorefSamples, concat the values key by key.

        :param: Some CorefSamples
        :return ~textflint.input.component.sample.CorefSample:
            A CorefSample, as the docs are concanated to form one x

        """
        if len(args) == 0:
            return CorefSample(data=None, origin=None)
        ret_coref_sample = CorefSample(data=None, origin=args[0])
        shift = 0
        for corefsam in args:
            if corefsam.doc_key != None:
                ret_coref_sample.doc_key = corefsam.doc_key
                break
        mask, x, sen_map, doc_sp = [], [], [], []
        clusters, constituents, ner = [], [], []

        for corefsam in args:
            @shift_decor
            def index_shift(word_idx):
                return word_idx + shift
            mask.extend(corefsam.x.mask)
            x.extend(corefsam.x.words)
            sen_map.extend(corefsam.sen_map)
            doc_sp.extend(corefsam.doc_sp.words)
            clusters.extend(
                recur_ap(index_shift, corefsam.clusters.field_value))
            constituents.extend(
                recur_ap(index_shift, corefsam.constituents.field_value))
            ner.extend(
                recur_ap(index_shift, corefsam.ner.field_value))
            shift += len(corefsam.x.words)

        ret_coref_sample.x = TextField(x, mask=mask)
        ret_coref_sample.sen_map = sen_map
        ret_coref_sample.doc_sp = TextField(doc_sp, mask=mask)
        ret_coref_sample.clusters = ListField(clusters)
        ret_coref_sample.constituents = ListField(constituents)
        ret_coref_sample.ner = ListField(ner)

        return ret_coref_sample

    def shuffle_conll(self, sen_idxs):
        r"""
        Given a CorefSample and shuffled sentence indexes, reproduce
        a CorefSample with respect to the indexes.

        :param list sen_idxs: a list of ints. the indexes in a shuffled order
                we expect `sen_idxs` is like [1, 3, 0, 4, 2, 5] when sen_num = 6
        :return ~textflint.input.component.sample.CorefSample:
            a CorefSample with respect to the shuffled index

        """
        # arg check
        assert len(sen_idxs) == self.num_sentences()
        assert len(set(sen_idxs)) == self.num_sentences()
        assert sum(sen_idxs) == self.num_sentences() * \
            (self.num_sentences() - 1) / 2
        # logic
        doc_key = self.doc_key
        # speakers, sentences
        sentences = CorefSample.doc2sens(self.x.words, self.sen_map)
        sens = [sentences[i] for i in sen_idxs]
        speakers = CorefSample.doc2sens(self.doc_sp.words, self.sen_map)
        sps = [speakers[i] for i in sen_idxs]
        # clusters, constituents, ner

        @shift_decor
        def index_shift(word_idx):
            # word_idx is in the sen_idx-th sentence
            sen_idx = self.index_in_sen(word_idx)
            ori_shift = sum(self.sen_map[:sen_idx])
            # shf_shift
            shf_sen_idx = sen_idxs.index(sen_idx)
            shf_shift = 0
            for j in range(shf_sen_idx):
                shf_shift += self.sen_map[sen_idxs[j]]
            return word_idx - (ori_shift - shf_shift)
        clusters = recur_ap(index_shift, self.clusters.field_value)
        constituents = recur_ap(index_shift, self.constituents.field_value)
        ner = recur_ap(index_shift, self.ner.field_value)
        ret_conll = {
            "doc_key": doc_key,
            "sentences": sens,
            "speakers": sps,
            "clusters": clusters,
            "constituents": constituents,
            "ner": ner
        }
        return CorefSample(data=ret_conll, origin=self)

    # methods for making coref part sample

    def part_conll(self, pres_idxs):
        r"""
        Only sentences with `indexs` will be kept, and all the structures of
        `clusters` are kept for convenience of concat.

        :param list pres_idxs: a list of ints. the indexes to be preserved
            we expect `pres_idxs` is from [0..num_sen], and is in ascending
            order, like [0, 1, 3, 5] when num_sen = 6
        :return ~textflint.input.component.sample.CorefSample:
            a CorefPartSample of a conll-part
        """
        # arg check
        sorted_idxs = sorted(pres_idxs)
        if len(sorted_idxs) >= 1:
            assert sorted_idxs[0] >= 0
            assert sorted_idxs[-1] < self.num_sentences()
        if len(sorted_idxs) >= 2:
            for i in range(len(sorted_idxs) - 1):
                assert sorted_idxs[i] < sorted_idxs[i+1]
        # main logic
        doc_key = self.doc_key
        # sentences, speakers
        sentences = CorefSample.doc2sens(self.x.words, self.sen_map)
        sens = [sentences[i] for i in pres_idxs]
        speakers = CorefSample.doc2sens(self.doc_sp.words, self.sen_map)
        sps = [speakers[i] for i in pres_idxs]
        # clusters, constituents, ner

        @shift_decor
        def index_shift(word_idx):
            # word_idx is in the sen_idx-th sentence
            sen_idx = self.index_in_sen(word_idx)
            ori_shift = sum(self.sen_map[:sen_idx])
            del_shift = sum([self.sen_map[i]
                             for i in pres_idxs if i < sen_idx])
            return word_idx - (ori_shift - del_shift)

        def preserve(span):
            return self.index_in_sen(span[0]) in pres_idxs
        clusters = recur_ap(
            index_shift,
            [[span for span in cluster if preserve(span)]
             for cluster in self.clusters.field_value])
        constituents = recur_ap(
            index_shift,
            [span for span in self.constituents.field_value if preserve(span)])
        ner = recur_ap(
            index_shift,
            [span for span in self.ner.field_value if preserve(span)])
        ret_conll = {
            "doc_key": doc_key,
            "sentences": sens,
            "speakers": sps,
            "clusters": clusters,
            "constituents": constituents,
            "ner": ner
        }
        return CorefPartSample(data=ret_conll, origin=self)

    def part_before_conll(self, sen_idx):
        r"""
        Only sentences [0, sen_idx) will be kept, and all the structures of
        `clusters` are kept for convenience of concat.

        :param int sen_idx: sentences with idx < sen_idx will be preserved
        :return ~textflint.input.component.sample.CorefSample:
            a CorefPartSample of a conll-part
        """
        doc_key = self.doc_key
        # sentences, speakers
        sentences = CorefSample.doc2sens(self.x.words, self.sen_map)
        sens = sentences[:sen_idx]
        speakers = CorefSample.doc2sens(self.doc_sp.words, self.sen_map)
        sps = speakers[:sen_idx]
        # clusters, constituents, ner

        @shift_decor
        def index_shift(word_idx):
            return word_idx

        def preserve(span):
            return self.index_in_sen(span[0]) < sen_idx
        clusters = recur_ap(
            index_shift,
            [[span for span in cluster if preserve(span)]
             for cluster in self.clusters.field_value])
        constituents = recur_ap(
            index_shift,
            [span for span in self.constituents.field_value if preserve(span)])
        ner = recur_ap(
            index_shift,
            [span for span in self.ner.field_value if preserve(span)])
        ret_conll = {
            "doc_key": doc_key,
            "sentences": sens,
            "speakers": sps,
            "clusters": clusters,
            "constituents": constituents,
            "ner": ner
        }
        return CorefPartSample(data=ret_conll, origin=self)

    def part_after_conll(self, sen_idx):
        r"""
        Only sentences [sen_idx:] will be kept, and all the structures of
        `clusters` are kept for convenience of concat.

        :param int sen_idx: sentences with idx < sen_idx will be preserved
        :return ~textflint.input.component.sample.CorefSample:
            a CorefPartSample of a conll-part

        """
        doc_key = self.doc_key
        # sentences, speakers
        sentences = CorefSample.doc2sens(self.x.words, self.sen_map)
        sens = sentences[sen_idx:]
        speakers = CorefSample.doc2sens(self.doc_sp.words, self.sen_map)
        sps = speakers[sen_idx:]
        # clusters, constituents, ner

        @shift_decor
        def index_shift(word_idx):
            return word_idx - sum(self.sen_map[:sen_idx])

        def preserve(span):
            return self.index_in_sen(span[0]) >= sen_idx
        clusters = recur_ap(
            index_shift,
            [[span for span in cluster if preserve(span)]
             for cluster in self.clusters.field_value])
        constituents = recur_ap(
            index_shift,
            [span for span in self.constituents.field_value if preserve(span)])
        ner = recur_ap(
            index_shift,
            [span for span in self.ner.field_value if preserve(span)])
        ret_conll = {
            "doc_key": doc_key,
            "sentences": sens,
            "speakers": sps,
            "clusters": clusters,
            "constituents": constituents,
            "ner": ner
        }
        return CorefPartSample(data=ret_conll, origin=self)


class CorefPartSample(CorefSample):
    r"""
    Coref Part Sample: corresponed to a part of a Coref Sample

    """

    def __init__(self, data, origin=None, sample_id=None):
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'CorefPartSample'

    def check_data(self, data):
        r"""
        Check if `data` is a conll-part. The condition is looser than conll

        :param None|dict data:
            Must have key: sentences, clusters
            May have key: doc_key, speakers, constituents, ner
        :return:

        """
        if data == None:
            return
        assert isinstance(data, dict), "To be loaded by CorefSample: not a dict"
        # doc_key: string
        if "doc_key" in data:
            assert isinstance(data["doc_key"], str), \
                "To be loaded by CorefSample: `doc_key` is not a str"

        # sentences: 2nd list of str; word list list
        assert "sentences" in data and isinstance(data["sentences"], list), \
            "To be loaded by CorefSample: `sentences` is not a list"
        if len(data["sentences"]) > 0:
            assert isinstance(data["sentences"][0], list), \
                "To be loaded by CorefSample: `sentences` is not a 2nd list"
            assert isinstance(data["sentences"][0][0], str), \
                "To be loaded by CorefSample: " \
                "`sentences` is not a word list list"

        # speakers: 2nd list of str; word list list
        if "speakers" in data:
            assert isinstance(data["speakers"], list), \
                "To be loaded by CorefSample: `speakers` is not a list"
            if len(data["speakers"]) > 0:
                assert isinstance(data["speakers"][0], list), \
                    "To be loaded by CorefSample: `speakers` is not a 2nd list"
                assert isinstance(data["speakers"][0][0], str), \
                    "To be loaded by CorefSample: " \
                    "`speakers` is not a word list list"

        # clusters: 2nd list of span([int, int]); cluster list
        assert "clusters" in data and isinstance(data["clusters"], list), \
            "To be loaded by CorefSample: `clusters` is not a list"
        if len(data["clusters"]) > 0:
            for cluster in data["clusters"]:
                assert isinstance(cluster, list), \
                    "To be loaded by CorefSample: " \
                    "cluster in `clusters` is not a list"
                if len(cluster) > 0:
                    assert isinstance(cluster[0][0], int), \
                        "To be loaded by CorefSample: " \
                        "cluster in `clusters` is not a span list"

        # constituents: list of tag([int, int, str])
        if "constituents" in data:
            assert isinstance(data["constituents"], list), \
                "To be loaded by CorefSample: `constituents` is not a list"
            if len(data["constituents"]) > 0:
                assert isinstance(data["constituents"][0], list), \
                    "To be loaded by CorefSample: " \
                    "constituent in `constituents` is not a list"
                assert isinstance(data["constituents"][0][0], int), \
                    "To be loaded by CorefSample: " \
                    "constituent in `constituents` is not a [b, e, label]"
                assert isinstance(data["constituents"][0][2], str), \
                    "To be loaded by CorefSample: " \
                    "constituent in `constituents` is not a [b, e, label]"

        # ner: list of tag([int, int, str])
        if "ner" in data:
            assert isinstance(data["ner"], list), \
                "To be loaded by CorefSample: `ner` is not a list"
            if len(data["ner"]) > 0:
                assert isinstance(data["ner"][0], list), \
                    "To be loaded by CorefSample: " \
                    "constituent in `ner` is not a list"
                assert isinstance(data["ner"][0][0], int), \
                    "To be loaded by CorefSample: " \
                    "constituent in `ner` is not a [b, e, label]"
                assert isinstance(data["ner"][0][2], str), \
                    "To be loaded by CorefSample: " \
                    "constituent in `ner` is not a [b, e, label]"

    # useful methods for making coref sample
    def remove_invalid_corefs_from_part(self):
        r"""
        conll parts may contain clusters that has only
        0 or 1 span, which is not a valid one.

        This function remove these invalid clusters from self.clusters.

        :return ~textflint.input.component.sample.CorefSample:
            a CorefSample that passes check_data

        """
        conll = self.dump()
        conll["clusters"] = [
            cluster for cluster in conll["clusters"] if len(cluster) > 1]
        return CorefSample(conll, origin=self)

    # other useful methods

    @staticmethod
    def concat_conll_parts(*args):
        r"""
        concat conll parts

        :param: many CorefPartSamples
            elements in which are assumed to be parts from the same conll,
            generated by part_conll.
            Merge result is still treated as a conll part, which should be
            postprocessed by `remove_invalid_corefs_from_part` to form a
            valid CorefSample.
        :return ~textflint.input.component.sample.CorefPartSample:
            a CorefPartSample of a conll-part

        """
        if len(args) == 0:
            return CorefPartSample(data=None, origin=None)
        ret_coref_sample = CorefPartSample(None, origin=args[0])
        shift = 0
        for corefsam in args:
            if corefsam.doc_key != None:
                ret_coref_sample.doc_key = corefsam.doc_key
                break
        mask, x, sen_map, doc_sp = [], [], [], []
        clusters = [[] for cluster in args[0].clusters.field_value]
        constituents, ner = [], []
        for corefsam in args:
            @shift_decor
            def index_shift(word_idx):
                return word_idx + shift
            mask.extend(corefsam.x.mask)
            x.extend(corefsam.x.words)
            sen_map.extend(corefsam.sen_map)
            doc_sp.extend(corefsam.doc_sp.words)
            samp_clusters = recur_ap(
                index_shift, corefsam.clusters.field_value)
            for i in range(len(clusters)):
                clusters[i].extend(samp_clusters[i])
            constituents.extend(
                recur_ap(index_shift, corefsam.constituents.field_value))
            ner.extend(
                recur_ap(index_shift, corefsam.ner.field_value))
            shift += len(corefsam.x.words)
        ret_coref_sample.x = TextField(x, mask=mask)
        ret_coref_sample.sen_map = sen_map
        ret_coref_sample.doc_sp = TextField(doc_sp, mask=mask)
        ret_coref_sample.clusters = ListField(clusters)
        ret_coref_sample.constituents = ListField(constituents)
        ret_coref_sample.ner = ListField(ner)
        return ret_coref_sample
