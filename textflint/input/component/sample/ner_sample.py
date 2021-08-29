r"""
NER Sample class to hold the necessary info and provide atomic operations.
==========================================================
"""
__all__ = ["NERSample"]
from .sample import Sample
from ..field import ListField, TextField
from ....common.utils.list_op import get_align_seq


class NERSample(Sample):
    r"""
    NER Sample class to hold the necessary info and provide atomic operations.

    """
    def __init__(
        self,
        data,
        origin=None,
        sample_id=None,
        mode='BIO'
    ):
        r"""
        :param dict data: The dict obj that contains data info
        :param ~BaseSample origin: Original sample obj
        :param int sample_id: the id of sample
        :param str mode: The sequence labeling mode for NER samples.
        """
        self.mode = mode
        self.text = None
        self.tags = None
        self.entities = None
        super().__init__(data, origin=origin, sample_id=sample_id)

    def __repr__(self):
        return 'NERSample'

    def check_data(self, data):
        r"""
        Check rare data format.

        :param dict data: rare data input.

        """
        assert 'x' in data and isinstance(data['x'], (str, list)), \
            r"Type of 'x' should be 'str' or 'list'"

        assert 'y' in data and isinstance(data['y'], (str, list)), \
            r"Type of 'y' should be 'list'"

        assert self.mode == 'BIO' or self.mode == 'BIOES', \
            'Not support {0} type, plz ensure mode in {1}' .format(
                    self.mode, ['BIO', 'BIOES'])

    def is_legal(self):
        if len(self.text.words) != len(self.tags):
            print('here')
            return False
        return True

    def load(self, data):
        r"""
        Parse data into sample field value.

        :param dict data: rare data input.

        """
        self.text = TextField(data['x'])
        tags = data['y'].split() if isinstance(data['y'], str) else data['y']
        self.tags = ListField(tags)

        # set mask to prevent UT transform modify entity word.
        if self.mode == 'BIO':
            self.entities = ListField(self.find_entities_BIO(
                self.text.words, self.tags))
        elif self.mode == 'BIOES':
            self.entities = ListField(self.find_entities_BIOES(
                self.text.words, self.tags))
        if not self.is_legal():
            raise ValueError('A failed transformation which leads to '
                             'mismatch between input and output.')

    def dump(self):
        r"""
        Convert sample info to input data json format.

        :return json: the dict of sentences and labels

        """
        if not self.is_legal():
            raise ValueError('A failed transformation which leads to '
                             'mismatch between input and output.')

        return {'x': self.text.words,
                'y': self.tags.field_value,
                'sample_id': self.sample_id}

    def delete_field_at_indices(self, field, indices):
        r"""
        Delete tokens and their NER tag.

        :param str field: field str
        :param list indices: list of int/list/slice
                shape：indices_num
                each index can be int indicate delete single item or
                    their list like [1, 2, 3],
                can be list like (0,3) indicate replace items
                    from 0 to 3(not included),
                can be slice which would be convert to list
        :return: Modified NERSample.

        """
        assert field == 'text'
        sample = self.clone(self)
        sample = super(NERSample, sample)\
            .delete_field_at_indices(field, indices)
        sample = super(NERSample, sample)\
            .delete_field_at_indices('tags', indices)

        return sample

    def delete_field_at_index(self, field, index):
        r"""
        Delete tokens and their NER tag.

        :param str field: field string, normally 'x'
        :param int|list|slice index: int/list/slice
                can be int indicate delete single item
                    or their list like [1, 2, 3],
                can be list like (0,3) indicate replace items
                    from 0 to 3(not included),
                can be slice which would be convert to list
        :return: Modified NERSample

        """
        return self.delete_field_at_indices(field, [index])

    def insert_field_before_indices(self, field, indices, items):
        r"""
        Insert tokens and ner tags.Assuming the tag of new_item is O.

        :param str field:field string
        :param list indices: list of int
                shape：indices_num, list like [1, 2, 3]
        :param list items: list of str/list
                shape: indices_num, correspond to indices
        :return: Modified NERSample

        """
        assert field == 'text'
        sample = self.clone(self)
        sample = super(NERSample, sample)\
            .insert_field_before_indices(field, indices, items)
        # add 'O' tag to insert token
        insert_tags = get_align_seq(items, 'O')
        sample = super(NERSample, sample)\
            .insert_field_before_indices('tags', indices, insert_tags)

        return sample

    def insert_field_before_index(self, field, ins_index, new_item):
        r"""
        Insert tokens and ner tags.Assuming the tag of new_item is O

        :param str field: field str
        :param int ins_index: indicate which index to insert items
        :param str/list new_item: items to insert
        :return: Modified NERSample

        """
        return self.insert_field_before_indices(field, [ins_index], [new_item])

    def insert_field_after_indices(self, field, indices, items):
        r"""
        Insert tokens and ner tags.Assuming the tag of new_item is O.

        :param str field: field string
        :param list indices: list of int
                shape：indices_num, like [1, 2, 3]
        :param list items: list of str/list
                shape: indices_num, correspond to indices
        :return: Modified NERSample

        """
        assert field == 'text'
        sample = self.clone(self)
        sample = super(NERSample, sample)\
            .insert_field_after_indices(field, indices, items)
        # add 'O' tag to insert token
        insert_tags = get_align_seq(items, 'O')
        sample = super(NERSample, sample)\
            .insert_field_after_indices('tags', indices, insert_tags)

        return sample

    def insert_field_after_index(self, field, ins_index, new_item):
        r"""
        Insert tokens and ner tags.Assuming the tag of new_item is O.

        :param str field: field string
        :param int ins_index: indicate where to apply insert
        :param str|list new_item: shape: indices_num,
            correspond to field_sub_items
        :return: Modified NERSample

        """
        return self.insert_field_after_indices(field, [ins_index], [new_item])

    def find_entities_BIO(self, word_seq, tag_seq):
        r"""
        find entities in a sentence with BIO labels.

        :param list word_seq: a list of tokens representing a sentence
        :param list tag_seq: a list of tags representing a tag sequence
            labeling the sentence
        :return list entity_in_seq: a list of entities found in the sequence,
                including the information of the start position & end position
                in the sentence, the category, and the entity itself.

        """
        entity_in_seq = []
        entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
        temp_entity = ""

        for i in range(len(word_seq)):
            assert tag_seq[i][0] in ['B', 'I', 'O'], \
                'entity labels should be started with \'B\' or \'I\' or \'O\'.'
            if tag_seq[i][0] == 'B':
                assert tag_seq[i][1] == '-', \
                    'entity labels should be like the format \'X-XXX\'.'
                entity['start'] = i
                entity['tag'] = tag_seq[i][2:]
                temp_entity = word_seq[i]
                if i == len(word_seq) - 1:
                    entity['end'] = i
                    entity['entity'] = temp_entity
                    entity_in_seq.append(entity)
                    entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
            elif tag_seq[i][0] == 'I':
                assert temp_entity != '', \
                    '\'I\' label cannot be the start of the entity.'
                assert tag_seq[i][1] == '-', \
                    'entity labels should be like the format \'X-XXX\'.'
                temp_entity += ' ' + word_seq[i]
                if i == len(word_seq) - 1:
                    entity['end'] = i
                    entity['entity'] = temp_entity
                    entity_in_seq.append(entity)
                    entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
            elif tag_seq[i] == 'O':
                if i > 0 and not tag_seq[i - 1] == 'O':
                    entity['end'] = i - 1
                    entity['entity'] = temp_entity
                    entity_in_seq.append(entity)
                    entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
                temp_entity = ''

        return entity_in_seq

    def find_entities_BIOES(self, word_seq, tag_seq):
        r"""
        find entities in a sentence with BIOES labels.

        :param list word_seq: a list of tokens representing a sentence
        :param list tag_seq: a list of tags representing a tag sequence
            labeling the sentence
        :return list entity_in_seq: a list of entities found in the sequence,
                including the information of the start position & end position
                in the sentence, the category, and the entity itself.

        """
        entity_in_seq = []
        entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
        temp_entity = ""

        for i in range(len(word_seq)):
            assert tag_seq[i][0] in ['B', 'I', 'O', 'E', 'S'], \
                'entity labels should be started with ' \
                '\'B\' or \'I\' or \'O\' or \'E\' or \'S\'.'
            if not tag_seq[i] == 'O':
                assert tag_seq[i][1] == '-', \
                    'entity labels should be like the format \'X-XXX\'.'
            if tag_seq[i][0] == 'B':
                assert temp_entity == '', \
                    '\'B\' label must be the start of the entity.'
                entity['start'] = i
                entity['tag'] = tag_seq[i][2:]
                temp_entity = word_seq[i]
            elif tag_seq[i][0] == 'I':
                assert temp_entity != '', \
                    '\'I\' label cannot be the start of the entity.'
                temp_entity += ' ' + word_seq[i]
            elif tag_seq[i][0] == 'E':
                assert temp_entity != '', \
                    '\'E\' label cannot be the start of the entity.'
                temp_entity += ' ' + word_seq[i]
                entity['end'] = i
                entity['entity'] = temp_entity
                entity_in_seq.append(entity)
                entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
                temp_entity = ''
            elif tag_seq[i][0] == 'S':
                assert temp_entity == '', \
                    '\'S\' label must be the start of the entity.'
                entity['start'] = i
                entity['end'] = i
                entity['entity'] = word_seq[i]
                entity_in_seq.append(entity)
                entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
                temp_entity = ''

        return entity_in_seq

    def entities_replace(self, entities_info, candidates):
        r"""
        Replace multi entity in once time.Assume input entities
        with reversed sequential.

        :param list entities_info: list of entity_info
        :param list candidates: candidate entities
        :return: Modified NERSample

        """
        assert len(entities_info) == len(candidates)
        assert isinstance(entities_info, list)
        assert isinstance(candidates, list)
        sample = self.clone(self)

        for i in range(len(entities_info)):
            entity = entities_info[i]
            sample = sample.entity_replace(entity['start'], entity['end'],
                                           candidates[i], entity['tag'])

        return sample

    def entity_replace(self, start, end, entity, label):
        r"""
        Replace one entity and update entities info.

        :param int start: the start position of the entity to be replaced
        :param int end: the end position of the entity to be replaced
        :param str entity: the entity to be replaced with
        :param str label: the category of the entity
        :return: Modified NERSample

        """
        assert start <= end, "start is before end!"
        sample = self.clone(self)
        entity = entity.split(" ")
        word_prefix = sample.text.words[:start]
        word_suffix = [] if end == len(sample.text.words) - 1 \
            else sample.text.words[end + 1:]
        sample.text = TextField(word_prefix + entity + word_suffix)
        tag_prefix = sample.tags[:start]
        tag_suffix = [] if end == len(sample.tags) - 1 \
            else sample.tags[end + 1:]

        if self.mode == 'BIO':
            sample.tags = ListField(tag_prefix + ["B-" + label]
                                    + ["I-" + label] * (len(entity) - 1)
                                    + tag_suffix)
        else:
            len_entity = len(entity)
            if len_entity == 1:
                substitude = ["S-" + label]
            elif len_entity == 2:
                substitude = ["B-" + label] + ["E-" + label]
            else:
                substitude = ["B-" + label] \
                             + ["I-" + label] * (len_entity - 2)  \
                             + ["E-" + label]
            sample.tags = ListField(tag_prefix + substitude + tag_suffix)
            # set mask for changed entities

        # for i in range(start, end+1):
        #     sample.text.set_mask(i, MODIFIED_MASK)
        sample.entities = ListField(
            sample.find_entities_BIO(sample.text.words, sample.tags))

        return sample


