"""
WSD Sample Class
============================================

"""
from .sample import Sample
from ..field import TextField, ListField
import json
from ....common.settings import TASK_MASK, PUNC
import nltk

# nltk.download('averaged_perceptron_tagger')
__all__ = ['WSDSample']


class WSDSample(Sample):
    r"""
        WSDSample Class

    """

    def __init__(
            self,
            data,
            origin=None,
            sample_id=None):
        """

        :param data: json type-sentence,pos,lemma,instance,sentence_id,source
        :param origin:
        :param sample_id:
        """
        self.sentence = None
        self.WSD_CORPUS = ['senseval2', 'senseval3', 'semeval2007',
                           'semeval2013', 'semeval2015', 'ALL']
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'WSDSample'

    def check_data(self, data):
        r"""
         Check the format of input data(json format).

          :param dict data: data name
          """
        # check sentence field
        assert 'sentence' in data and \
               isinstance(data['sentence'], list) and \
               len(data['sentence']) > 0, \
            "Sentence should be in data, and type of sentence should be list."
        for word in data['sentence']:
            assert isinstance(word, str), \
                "Type of item in term_list should be str."
        # check pos field
        assert 'pos' in data and isinstance(data['pos'], list), \
            "Pos should be in data"
        assert len(data['pos']) == len(data['sentence']), \
            "Type of pos should be list of sentence length."
        for pos in data['pos']:
            assert isinstance(pos, str), \
                "Type of item in pos should be str."
        # check lemma field
        assert 'lemma' in data and isinstance(data['lemma'], list) and \
               len(data['lemma']) == len(data['sentence']), \
            "Lemma should be in data, and type of lemma should be list of sentence length."
        for lemma in data['lemma']:
            assert isinstance(lemma, str), \
                "Type of item in lemma should be str."
        # check instance field
        assert 'instance' in data and isinstance(data['instance'], list) and \
               len(data['instance']) > 0, \
            "Instance should be in data, and type of instance should be list."
        # check every single instance
        for key, start, end, target, gk in data['instance']:
            assert isinstance(key, str) and isinstance(start, int) and \
                   isinstance(end, int) and isinstance(target, str) and \
                   isinstance(gk, str), \
                "Type of item in instance should be [str,int,int,str,str]"
            assert start < end, 'End position should be after start position.'
            assert ' '.join(
                data['sentence'][start:end]).lower() == target.lower(), \
                'Target word is not at the correct position.'
        # check sentence_id field
        assert 'sentence_id' in data and isinstance(data['sentence_id'],
                                                    str) and \
               len(data['sentence_id']) > 0, \
            "Sentence_id should be in data, and type of sentence_id should be a valid str."
        # check source field
        assert 'source' in data and isinstance(data['source'], str) and \
               data['source'] in self.WSD_CORPUS, \
            "Source should be in data, and type of source should be a valid str."

    def is_legal(self):
        r"""
        Validate whether the sample is legal
        :return bool
        """
        # get attribute value
        sentence = self.sentence.field_value
        instance = self.instance.field_value
        # check instance:
        if len(instance) > 0:
            for key, start, end, word, _ in instance:
                if key is None or key == "":
                    return False
                if start >= end:
                    return False
                # check target word
                if word.lower() != " ".join(sentence[start:end]).lower():
                    return False
        return True

    def get_idx_list(self, indices):
        r"""
        Get index list from indices
        :param list indices: a list of index varying in type(int,list,slice)
        :return: list idx_list:a list of index(int)
        """
        idx_list = []
        # check validity
        for span in indices:
            assert isinstance(span, (int, list, slice))
            if isinstance(span, int):
                assert span >= 0, 'invalid indices'
            if isinstance(span, list):
                assert len(span) == 2, 'invalid indices'
            if isinstance(span, slice):
                assert span.start >= 0, 'invalid indices'
                assert span.stop >= span.start, 'invalid indices'
                assert span.step == 1 or span.step is None, 'invalid indices'
        for span in indices:
            if isinstance(span, int):
                idx_list.append(span)
            elif isinstance(span, list):
                idx_list.extend(range(span[0], span[1]))
            elif isinstance(span, slice):
                idx_list.extend(range(span.stop)[span])
        # get all positions according to the mixed list
        idx_list = sorted(list(set(idx_list)))
        return idx_list

    def check_indices_and_items(self, indices, items):
        r"""
        check validity of indices
        :param list indices: a list of index varying in type(int,list,slice)
        :param bool equal: default False, indicates whether the replacement is eaual
        :param list items:
        """
        assert len(indices) > 0, 'indices should be not null!'
        assert len(indices) == len(items), 'illegal insertion!'
        # check validity of index
        for span, word_list in zip(indices, items):
            assert isinstance(span, int) or isinstance(span,
                                                       list) or isinstance(span,
                                                                           slice)
            if isinstance(span, int):
                assert span >= 0, 'invalid indices'
            elif isinstance(span, list):
                assert len(span) == 2, 'invalid indices'
            elif isinstance(span, slice):
                assert span.start >= 0, 'invalid indices'
                assert span.stop >= span.start, 'invalid indices'
                assert span.step == 1 or span.step is None, 'invalid indices'
            else:
                raise AssertionError

    def load(self, data):
        r"""
         Load the legal data and convert it into WSDSample.

         :param dict data: data name
         """
        self.sentence = TextField(data['sentence'])
        self.lemma = ListField(data['lemma'])
        self.pos = ListField(data['pos'])
        self.instance = ListField(data['instance'])
        self.sentence_id = TextField(data['sentence_id'])
        self.source = TextField(data['source'])
        if not self.is_legal():
            raise ValueError("Data is not legal to load")
        # set mask
        for key, start, end, word, _ in self.instance.field_value:
            for i in range(start, end):
                self.sentence.set_mask(i, TASK_MASK)

    def dump(self):
        r"""
        Dump the legal data in json format.
        :return: dict:sample in json format
        """
        # check legality before dump
        if not self.is_legal():
            raise ValueError("Data is not legal to dump")
        # fetch data of json format
        return {
            'sentence': self.sentence.field_value,
            'pos': self.pos.field_value,
            'lemma': self.lemma.field_value,
            'instance': self.instance.field_value,
            'sentence_id': self.sentence_id.field_value,
            'source': self.source.field_value,
            'sample_id': self.sample_id
        }

    def tag_words(self, words, wn=False):
        r"""
        Get pos tag for list of words.
        :param words: a list of words to be tagged,e.g.['he',['pointed','out']]
        :param wn: default false;true if get wordnet-level pos tag
        :return: tagged_sent: pos tag for list of words
        """

        def normalize_pos(pos):
            r"""
            Convert pos tag into wordnet-level pos tag
            :param pos: original pos tag
            :return: pp: wordnet-level pos tag
            """
            if pos in ["a", "r", "n", "v", "s"]:
                pp = pos
            else:
                norm_dict = {"JJ": "a", "VB": "v", "NN": "n", "RB": "r"}
                key = pos[:2]
                if key in norm_dict.keys():
                    pp = norm_dict[key]
                else:
                    pp = None
            return pp

        def convert_pos(pos):
            r"""
            Convert pos tag into pos tag applied in xml file.
            :param pos:original pos tag
            :return:pp: pos tag applied in xml file
            """
            if pos in PUNC:
                pp = '.'
            else:
                convert_dict = {"JJ": "ADJ", "VB": "VERB", "NN": "NOUN",
                                "RB": "ADV", "CD": "NUM", "DT": "DET",
                                "IN": "ADP", "CC": "CONJ", "POS": "PRT",
                                "PRP": "PRON"}
                key2 = pos[:2]
                key3 = pos[:3]
                if key2 in convert_dict.keys():
                    pp = convert_dict[key2]
                elif key3 in convert_dict.keys():
                    pp = convert_dict[key3]
                else:
                    pp = "X"
            return pp

        tagged_sent = []
        for item in words:
            assert isinstance(item, str) or isinstance(item,
                                                       list), 'invalid format of list'
            if isinstance(item, str):
                if wn is True:
                    tagged_sent.append(
                        normalize_pos(nltk.pos_tag([item])[0][1]))
                else:
                    if item in PUNC:
                        tagged_sent.append('.')
                    else:
                        tagged_sent.append(
                            convert_pos(nltk.pos_tag([item])[0][1]))
            elif isinstance(item, list):
                for word in item:
                    assert isinstance(word, str), 'invalid format of word'
                pos_list = [t[1] for t in nltk.pos_tag(item)]
                for i in range(len(pos_list)):
                    if item[i] in PUNC:
                        pos_list[i] = '.'
                if wn is True:
                    tagged_sent.append([normalize_pos(t) for t in pos_list])
                else:
                    tagged_sent.append([convert_pos(t) for t in pos_list])

        return tagged_sent

    def get_lemma(self, words):
        r"""
        Get lemma for list of words.
        :param words: a list of words,e.g.['he',['pointed','out']]
        :return: lemma_sent: lemma for list of words
        """
        from nltk.stem import WordNetLemmatizer
        wnl = WordNetLemmatizer()
        lemma_sent = []
        # get corresponding pos tag according to wordnet
        pos = self.tag_words(words, wn=True)
        for word, tag in zip(words, pos):
            if isinstance(word, str):
                if isinstance(tag, str):
                    lemma_sent.append(wnl.lemmatize(word, tag))
                else:
                    lemma_sent.append(word)  # None type
            elif isinstance(word, list):
                assert isinstance(tag, list), 'invalid format of tag'
                assert len(word) == len(tag), 'inconsistent tagging'
                lemma = []
                for w, t in zip(word, tag):
                    assert isinstance(w, str), 'invalid format of word'
                    if isinstance(t, str):
                        lemma.append(wnl.lemmatize(w, t))
                    else:
                        lemma.append(w)
                lemma_sent.append(lemma)
        return lemma_sent

    def delete_field_at_indices(self, field, indices):
        r"""delete word of given indices in sentence

        :param string field: field to be operated on
        :param list indices: a list of index to be deleted

        :return WSDSample sample: a modified sample
        """
        assert len(indices) > 0, 'indices should be not null!'
        sample = self.clone(self)
        sample = super(
            WSDSample,
            sample).delete_field_at_indices(
            field,
            indices)
        # delete corresponding pos
        sample = super(
            WSDSample,
            sample).delete_field_at_indices(
            'pos',
            indices)
        # delete corresponding lemma
        sample = super(
            WSDSample,
            sample).delete_field_at_indices(
            'lemma',
            indices)
        # adjust positions of corresponding instance/delete the exact instance
        # check if any target word is deleted
        idx_list = self.get_idx_list(indices)
        targets = []  # update instance information
        for key, start, end, word, sense in sample.instance:
            # if deleted,skip this target word
            if start in idx_list or end - 1 in idx_list:
                continue
            # add intact target
            targets.append([key, start, end, word, sense])

        # get new position for each instance
        for i in range(len(targets)):
            # targets are kept impact
            start = targets[i][1]
            offset = len([idx for idx in idx_list if idx < start])
            targets[i][1] -= offset  # start
            targets[i][2] -= offset  # end
        setattr(sample, 'instance', ListField(targets))
        return sample

    def delete_field_at_index(self, field, del_index):
        """ Delete items of given scopes of field value.

        :param string field: transformed field
        :param list del_index: index of delete position
        :return WSDSample sample: a modified sample
        """
        return self.delete_field_at_indices(field, [del_index])

    def insert_field_after_indices(self, field, indices, items, pos_list=None,
                                   lemma_list=None):
        r"""insert word after given indices in sentence

        :param string field: field to be operated on
        :param list indices: a list of index to be inserted
        :param list items: list of items to be inserted
        :param list pos_list: default none, list of given pos for items
        :param list lemma_list: default none, list of given lemma for items
        :return WSDSample sample: a modified sample
        """
        self.check_indices_and_items(indices, items)
        sample = self.clone(self)
        sample = super(WSDSample, sample).insert_field_after_indices(field,
                                                                     indices,
                                                                     items)

        # insert corresponding pos
        if pos_list is None:
            new_pos = sample.tag_words(items)
        else:
            new_pos = pos_list
        sample = super(WSDSample, sample).insert_field_after_indices('pos',
                                                                     indices,
                                                                     new_pos)

        # insert corresponding lemma
        if lemma_list is None:
            new_lemma = sample.get_lemma(items)
        else:
            new_lemma = lemma_list
        sample = super(WSDSample, sample).insert_field_after_indices('lemma',
                                                                     indices,
                                                                     new_lemma)

        # adjust the position of the influenced instances
        targets = []
        for key, start, end, word, sense in sample.instance:
            offset_s = 0
            for i, idx in enumerate(indices):
                if idx < start:
                    if isinstance(items[i], str):
                        offset_s += 1
                    else:
                        offset_s += len(items[i])
                else:
                    break
            offset_e = offset_s
            targets.append([key, start + offset_s, end + offset_e, word, sense])
        # reconstruct instances
        setattr(sample, 'instance', ListField(targets))
        return sample

    def insert_field_after_index(self, field, ins_index, new_item,
                                 pos_list=None, lemma_list=None):
        r"""
        insert word after given index in sentence

        :param string field: field to be operated on
        :param int ins_index: indicate which index to insert items
        :param str/list new_item: items to insert
        :param list pos_list: default none, list of given pos for items
        :param list lemma_list: default none, list of given lemma for items
        :return: WSDSample: a modified sample
        """
        if pos_list is None and lemma_list is None:
            return self.insert_field_after_indices(field, [ins_index],
                                                   [new_item])
        elif pos_list is None:
            return self.insert_field_after_indices(field, [ins_index],
                                                   [new_item],
                                                   lemma_list=[lemma_list])
        elif lemma_list is None:
            return self.insert_field_after_indices(field, [ins_index],
                                                   [new_item],
                                                   pos_list=[pos_list])
        else:
            return self.insert_field_after_indices(field, [ins_index],
                                                   [new_item],
                                                   pos_list=[pos_list],
                                                   lemma_list=[lemma_list])

    def insert_field_before_indices(self, field, indices, items, pos_list=None,
                                    lemma_list=None):
        r"""insert word before given indices in sentence

        :param string field: field to be operated on
        :param list indices: a list of index to be inserted
        :param list items: list of items to be inserted
        :param list pos_list: default none, list of given pos for items
        :param list lemma_list: default none, list of given lemma for items
        :return WSDSample sample: a modified sample

        """
        self.check_indices_and_items(indices, items)
        sample = self.clone(self)
        sample = super(WSDSample, sample).insert_field_before_indices(field,
                                                                      indices,
                                                                      items)

        # insert corresponding pos
        if pos_list is None:
            new_pos = sample.tag_words(items)
        else:
            new_pos = pos_list
        sample = super(WSDSample, sample).insert_field_before_indices('pos',
                                                                      indices,
                                                                      new_pos)

        # insert corresponding lemma
        if lemma_list is None:
            new_lemma = sample.get_lemma(items)
        else:
            new_lemma = lemma_list
        sample = super(WSDSample, sample).insert_field_before_indices('lemma',
                                                                      indices,
                                                                      new_lemma)
        # adjust the position of the influenced instances
        targets = []
        for key, start, end, word, sense in sample.instance:
            offset_s = 0
            for i, idx in enumerate(indices):
                if idx <= start:
                    if isinstance(items[i], str):
                        offset_s += 1
                    else:
                        offset_s += len(items[i])
                else:
                    break
            # reconstruct instances
            offset_e = offset_s
            targets.append([key, start + offset_s, end + offset_e, word, sense])
        setattr(sample, 'instance', ListField(targets))
        return sample

    def insert_field_before_index(self, field, ins_index, new_item,
                                  pos_list=None, lemma_list=None):
        r"""
        insert word before given index in sentence

        :param string field: field to be operated on
        :param int ins_index: indicate which index to insert items
        :param str/list new_item: items to insert
        :param list pos_list: default none, list of given pos for items
        :param list lemma_list: default none, list of given lemma for items
        :return: WSDSample: a modified sample
        """
        if pos_list is None and lemma_list is None:
            return self.insert_field_before_indices(field, [ins_index],
                                                    [new_item])
        elif pos_list is None:
            return self.insert_field_before_indices(field, [ins_index],
                                                    [new_item],
                                                    lemma_list=[lemma_list])
        elif lemma_list is None:
            return self.insert_field_before_indices(field, [ins_index],
                                                    [new_item],
                                                    pos_list=[pos_list])
        else:
            return self.insert_field_before_indices(field, [ins_index],
                                                    [new_item],
                                                    pos_list=[pos_list],
                                                    lemma_list=[lemma_list])

    def pretty_print(self):
        data = self.dump()
        print(json.dumps(data))

    def replace_field_at_indices(self, field, indices, items, pos_list=None,
                                 lemma_list=None):
        r"""
        Replace scope items of field value with items of same length.
        :param str field: transformed field
        :param list indices: indices of delete positions
        :param list items: insert items
        :param list pos_list: default none, list of pos for items
        :param list lemma_list: default none, list of lemma for items
        :return WSDSample: a modified sample
        """
        self.check_indices_and_items(indices, items)
        # special for swap target
        if field == "sensekey":
            sensekeys = items
            items = list()
            for sk in sensekeys:
                items.append([t.split('%')[0] for t in sk])
        sample = self.clone(self)
        # replace word in sentences
        sample = super(WSDSample, sample).replace_field_at_indices('sentence',
                                                                   indices,
                                                                   items)
        # replace corresponding pos
        # pos won't change in swap target case
        if field != 'sensekey':
            new_pos = sample.tag_words(items)
            sample = super(WSDSample, sample).replace_field_at_indices('pos',
                                                                       indices,
                                                                       new_pos)

        # replace corresponding lemma
        sample = super(WSDSample, sample).replace_field_at_indices('lemma',
                                                                   indices,
                                                                   items)

        # check if any target word is replaced
        if field != "sensekey":
            idx_list = self.get_idx_list(indices)
        targets = []  # update instance information
        for key, start, end, word, sense in sample.instance:
            # not change target word
            if field != 'sensekey':
                # skip the replaced targets
                if start in idx_list or end - 1 in idx_list:
                    continue
                targets.append([key, start, end, word, sense])
            # swap target word
            else:
                # target word unchanged
                if [start, end] not in indices:
                    targets.append([key, start, end, word, sense])
                # target word changed
                else:
                    pos = indices.index([start, end])
                    new_sense = ' '.join(sensekeys[pos]).split(' ')[0]
                    new_word = ' '.join(items[pos])
                    targets.append([key, start, end, new_word, new_sense])
        # reconstruct instances
        setattr(sample, 'instance', ListField(targets))
        return sample

    def replace_field_at_index(self, field, index, items, pos_list=None,
                               lemma_list=None):
        r"""
        Replace scope items of field value with items.
        :param str field: transformed field
        :param list index: indices of delete positions
        :param list items: insert items
        :param list pos_list: default none, list of pos for items
        :param list lemma_list: default none, list of lemma for items
        :return WSDSample: a modified sample
        """
        if pos_list is None and lemma_list is None:
            return self.replace_field_at_index(field, [index],
                                               [items])
        elif pos_list is None:
            return self.replace_field_at_index(field, [index],
                                               [items],
                                               lemma_list=[lemma_list])
        elif lemma_list is None:
            return self.replace_field_at_index(field, [index],
                                               [items],
                                               pos_list=[pos_list])
        else:
            return self.replace_field_at_index(field, [index],
                                               [items],
                                               pos_list=[pos_list],
                                               lemma_list=[lemma_list])

    def unequal_replace_field_at_indices(self, field, indices, items):
        r"""
        Replace scope items of field value with items of different length.
        :param str field: transformed field
        :param list indices: indices of delete positions
        :param list items: insert items
        :return WSDSample: a modified sample
         """
        self.check_indices_and_items(indices, items)
        sample = self.clone(self)
        ori_instance = sample.instance.field_value
        targets = list()
        offset = 0
        for key, start, end, word, sense in ori_instance:
            # target word unchanged
            if [start, end] not in indices:
                targets.append(
                    [key, start + offset, end + offset, word, sense])
            else:
                # target word changed
                idx = indices.index([start, end])
                new_item = items[idx]  # sense key
                new_start = start + offset
                new_word = ' '.join([t.split('%')[0] for t in new_item])
                new_word = ' '.join(new_word.split('_'))
                offset += len(new_word.split(' ')) - (end - start)
                new_end = end + offset
                targets.append(
                    [key, new_start, new_end, new_word, new_item[0]])
        sorted_items, sorted_indices = zip(
            *sorted(zip(items, indices), key=lambda x: x[1], reverse=True))
        # record pos_list
        ori_pos = sample.pos.field_value
        pos_list = list()
        lemma_list = list()
        for index, item in zip(sorted_indices, sorted_items):
            start, end = index
            tag = ori_pos[start:end][0]
            ins_word = item[0].split('%')[0].split('_')
            pos_list.append([tag] * len(ins_word))
            lemma_list.append([item[0].split('%')[0]] * len(ins_word))
        for idx, item in enumerate(zip(sorted_items, pos_list, lemma_list)):
            sorted_token, sorted_pos, sorted_lemma = item
            ins_word = sorted_token[0].split('%')[0].split('_')
            sample = sample.delete_field_at_index(field,
                                                  sorted_indices[idx])
            insert_index = sorted_indices[idx] \
                if isinstance(sorted_indices[idx], int) \
                else sorted_indices[idx][0]
            field_obj = getattr(sample, field)
            if insert_index > len(field_obj):
                raise ValueError('Cant replace items at range {0}'
                                 .format(sorted_indices[idx]))
            elif insert_index == len(field_obj):
                sample = sample.insert_field_after_index(
                    field, insert_index - 1, ins_word, sorted_pos,
                    sorted_lemma)
            else:
                sample = sample.insert_field_before_index(
                    field, insert_index, ins_word, sorted_pos, sorted_lemma)

        setattr(sample, 'instance', ListField(targets))
        return sample
