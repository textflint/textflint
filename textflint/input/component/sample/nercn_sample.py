r"""
CNNER Sample class to hold the necessary info and provide atomic operations.
==========================================================
"""
__all__ = ["NERCnSample"]
from .cnsample import CnSample
from ..field import ListField, CnTextField
from ....common.utils.list_op import get_align_seq


class NERCnSample(CnSample):
    r"""
    CNNER Sample class to hold the necessary info and provide atomic operations.
    the input x can be a sentence or a list of single words,
    while y being a list of tags aligned to each word in x.

    Example::
        1. {'x':'上海浦东开发与法制建设同步', 'y':'['B-GPE','E-GPE','B-GPE','E-GPE','O','O','O','O','O','O','O','O','O']'}

        2. {
            'x':['上','海','浦','东','开','发','与','法','制','建','设','同','步'],
            'y':['B-GPE','E-GPE','B-GPE',E'-GPE','O','O','O','O','O','O','O','O','O']
           }

    """
    def __init__(
        self,
        data,
        origin=None,
        sample_id=None,
        mode='BMOES'
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
        return 'NERCnSample'

    def is_legal(self):
        if len(self.text.tokens) != len(self.tags):
            return False
        return True
    
    def check_data(self, data):
        r"""
        Check rare data format.
        1.  'x' in data and type(x) being str|list
        2.  'y' in data and type(y) being list
        3.  len(data['x']) == len(data['y'])
        4.  mode check
        5. check BMOES

        """
        assert 'x' in data and isinstance(data['x'], (str, list)), \
            r"Type of 'x' should be 'str' or 'list'"

        assert 'y' in data and isinstance(data['y'], (list)), \
            r"Type of 'y' should be 'list'"
        
        assert len(data['x'])==len(data['y']), \
            r'The number of words is not equal to the number of tags'
        
        assert self.mode == 'BMOES', \
            'Not support {0} type, plz ensure mode in {1}' .format(
                    self.mode, ['BMOES'])
        for tag in data['y']:

            assert tag[0] in ['B','M','O','E','S'],\
                'tags should start with B/M/O/E/S in mode BMOES'
            assert tag=='O' or (tag[0]!='O' and tag[1]=='-'),\
                'entity labels should be like the format \'X-XXX\'.'
        


    def load(self, data):
        r"""
        Parse data into sample field value.
        :param dict data: rare data input.

        """
        self.text = CnTextField(data['x'])
        tags = data['y']
        self.tags = ListField(tags)

        # set mask to prevent UT transform modify entity word.
        if self.mode == 'BMOES':
            self.entities = self.find_entities_BMOES(self.text.tokens, self.tags)



    def dump(self):
        r"""
        Convert sample info to input data json format.

        :return json: the dict of sentences and labels

        """

        return {'x': self.text.tokens,
                'y': self.tags.field_value,
                'sample_id': self.sample_id}



    def find_entities_BMOES(self, token_seq, tag_seq):
        r"""
        find entities in a sentence with BIOES labels.

        :param list token_seq: a list of tokens representing a sentence
        :param list tag_seq: a list of tags representing a tag sequence
            labeling each word of the sentence
        :return list entity_in_seq: a list of entities found in the sequence,
                including the information of the start position & end position
                in the sentence, the category, and the entity itself.

        """
        entity_in_seq = []
        entity = {'start': 0, 'end': 0, 'entity': "", 'tag': ""}
        temp_entity = ''
        for i in range(len(token_seq)):
            
            if tag_seq[i][0] == 'B':
                assert temp_entity == '', \
                    '\'B\' label must be the start of the entity.'
                entity['start'] = i
                entity['tag'] = tag_seq[i][2:]
                temp_entity = token_seq[i]
            elif tag_seq[i][0] == 'M':
                assert temp_entity != '', \
                    '\'M\' label cannot be the start of the entity.'
                temp_entity += token_seq[i]
            elif tag_seq[i][0] == 'E':
                assert temp_entity != '', \
                    '\'E\' label cannot be the start of the entity.'
                temp_entity += token_seq[i]
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
                entity['tag'] = tag_seq[i][2:]
                entity['entity'] = token_seq[i]
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
        assert len(entities_info) and len(entities_info) == len(candidates)
        assert isinstance(entities_info, list)
        assert isinstance(candidates, list)
        sample = self.clone(self)

        offset = 0
        for entity,candidate in zip(entities_info, candidates):
            
            rep_range = (entity['start']+offset, entity['end']+offset+1)
            candidate = list(candidate)
            
            len_candidate = len(candidate)
            offset += (len_candidate - len(entity['entity']))
            ent_label = entity['tag']

            if len_candidate==1:
                tags=['S-'+ent_label]
            elif len_candidate==2:
                tags=['B-'+ent_label, 'E-'+ent_label]
            else:
                tags= ['B-'+ent_label] + \
                    ['M-'+ent_label]*(len_candidate-2) + \
                    ['E-'+ent_label]
            sample = sample.unequal_replace_field_at_indices('text', [rep_range], [candidate])
            sample = sample.unequal_replace_field_at_indices('tags', [rep_range], [tags])

        sample.entities = sample.find_entities_BMOES(sample.text.tokens, sample.tags)
        return sample
