r"""
AddClause class for adding entity description transformation
"""
__all__ = ["InsertClause"]
import urllib.request
import wikidata
from wikidata.client import Client
import random
import json

from ....common.utils.error import FlintError
from ...transformation import Transformation
from ....input.component.sample.re_sample import RESample
from ....common.settings import WIKIDATA_STATEMENTS_no_zero, \
    CLAUSE_HEAD_no_zero, scbase


class InsertClause(Transformation):
    r"""
    Add extra entity-related clause to text

    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

    def __repr__(self):
        return 'InsertClause'

    def search_list(self, query):
        r"""Retrieve entity id from wikidata

        :param string query: name of query entity
        :return list: information of query entity
        """
        search = scbase + query
        try:
            url = urllib.request.urlopen(search)
        except OSError:
            raise FlintError('Time out to access Wikidata, '
                             'plz check your network!')
        else:
            data = json.loads(url.read().decode())
        return data['search']

    def _get_clause(self, entid):
        r"""
        Obtain entity-related clause

        :param string entid: index of entity
        :return string: entity clause
        """
        assert (isinstance(entid, str)), \
            f"the type of 'entid' should be str, got {type(entid)} instead"
        client = Client()
        entity = client.get(entid)

        random.shuffle(WIKIDATA_STATEMENTS_no_zero)
        for sta in WIKIDATA_STATEMENTS_no_zero:
            staent = client.get(sta)
            statement = entity.getlist(staent)
            if statement:
                if isinstance(statement[0], wikidata.entity.Entity):
                    ret = CLAUSE_HEAD_no_zero[sta] + str(statement[0].label)
                else:
                    ret = CLAUSE_HEAD_no_zero[sta] + str(statement[0])
                return ret
        return CLAUSE_HEAD_no_zero['description'] + str(entity.description)

    def get_clause(self, query):
        r"""
        obtain entity description

        :param string query: name of query entity
        :return string: entity description

        """
        assert (isinstance(query, str)), \
            f"the type of 'query' should be str, got {type(query)} instead"
        scrsl = self.search_list(query)
        clauseadd = ''
        if scrsl:
            entinf = scrsl[0]
            clauseadd = self._get_clause(entinf['id'])
        return clauseadd

    def _transform(self, sample, n=1, field='x', **kwargs):
        r"""
        Transform text string according to its entities

        :param RESample sample: sample input
        :param int n:  number of generated samples (no more than one)
        :return list: transformed sample list
        """
        assert(isinstance(sample, RESample)), \
            f"the type of 'sample' should be RESample, got " \
            f"{type(sample)} instead"
        assert(isinstance(n, int)), f"the type of 'n' should be int, " \
                                    f"got {type(n)} instead"

        sh, st, oh, ot = sample.get_en()
        text, relation = sample.get_sent()

        trans_sample = {}
        head_entity_span = [sh, st]
        head_entity_name = ' '.join(
            text[head_entity_span[0]:head_entity_span[1] + 1])
        tail_entity_span = [oh, ot]
        tail_entity_name = ' '.join(
            text[tail_entity_span[0]:tail_entity_span[1] + 1])
        new_text = text
        query = head_entity_name.replace(" ", "%20")
        clauseadd = self.get_clause(query).split()

        if clauseadd:
            if head_entity_span[1] < tail_entity_span[0]:  # head ... tail
                tail_entity_span[0], tail_entity_span[1] = \
                    tail_entity_span[0] + len(clauseadd), tail_entity_span[
                    1] + len(clauseadd)
            new_text = new_text[:head_entity_span[1] + 1] + \
                       clauseadd + new_text[head_entity_span[1] + 1:]
        query = tail_entity_name.replace(" ", "%20")
        clauseadd = self.get_clause(query).split()

        if clauseadd:
            if tail_entity_span[1] < head_entity_span[0]:  # tail ... head
                head_entity_span[0], head_entity_span[1] = \
                    head_entity_span[0] + len(clauseadd), \
                    head_entity_span[1] + len(clauseadd)
            new_text = new_text[:tail_entity_span[1] + 1] + \
                       clauseadd + new_text[tail_entity_span[1] + 1:]

        assert head_entity_name == " ".join(
            new_text[head_entity_span[0]:head_entity_span[1] + 1])
        assert tail_entity_name == " ".join(
            new_text[tail_entity_span[0]:tail_entity_span[1] + 1])

        trans_sample['x'], trans_sample['subj'], trans_sample['obj'], \
        trans_sample['y'] = new_text,\
                            head_entity_span, tail_entity_span, relation
        trans_samples = sample.replace_sample_fields(trans_sample)

        return [trans_samples]
