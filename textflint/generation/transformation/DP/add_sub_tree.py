r"""
Add a subtree in the sentence
============================================

"""

__all__ = ["AddSubTree"]

import json
import random
import urllib.request
from wikidata.client import Client
from wikidata.entity import Entity, EntityId
from ..transformation import Transformation
from ....common.utils.error import FlintError
from ....common.settings import WIKIDATA_STATEMENTS, \
    CLAUSE_HEAD, WIKIDATA_INSTANCE


class AddSubTree(Transformation):
    r"""
    Transforms the input sentence by adding a subordinate clause from WikiData.

    Example::
        original: "And it left mixed signals for London."
        transformed: "And it left mixed signals for London, which is a capital
            and largest city of the United Kingdom."

    """

    def __repr__(self):
        return 'AddSubtree'

    def _transform(self, sample, n=5, **kwargs):
        r"""
        Transform each sample case.

        :param ~DPSample sample:
        :return: transformed sample list.

        """
        entity_list = self.find_entity(sample)
        words = sample.get_words('x')
        deprels = sample.get_value('deprel')
        result = []

        for i, word_id in enumerate(entity_list):
            if i >= n:
                break
            else:
                word_list = [words[a - 1] for a in word_id]
                entity = '%20'.join(word_list)
                clause_add = self.get_clause(entity)

                if clause_add:
                    tokens_add = [','] + clause_add.split()
                    if deprels[word_id[-1]] != 'punct':
                        tokens_add.append(',')

                    for token in tokens_add[::-1]:
                        sample_mod = sample.insert_field_before_index(
                            'x', word_id[-1], token)
                    result.append(sample_mod)

        return result

    def search_list(self, query):
        r"""
        Search on Wikidata for the associated entries with the given query.

        :param str query: A list of words in the entity,
            which is joined by '%20'.
        :return: A list of the information of entries searched.

        """
        url_common = ('https://www.wikidata.org/w/api.php?action=wbsearch'
                      'entities&language=en&format=json&limit=3&search=')
        url_full = url_common + query
        try:
            url = urllib.request.urlopen(url_full)
        except OSError:
            raise FlintError('Time out to access Wikidata, '
                             'plz check your network!')
        else:
            data = json.loads(url.read().decode())
        return data['search']

    def clause_generate(self, entity_id):
        r"""
        Generate a subordinate clause for the given Wikidata entry.

        :param str entity_id: The ID of the given Wikidata entry.
        :return: The subordinate clause generated.

        """
        client = Client()
        entity = client.get(EntityId(entity_id))
        instance = client.get(EntityId(WIKIDATA_INSTANCE['instance']))
        instances = entity.getlist(instance)
        disamb = client.get(EntityId(WIKIDATA_INSTANCE['disambiguation']))
        if disamb in instances:
            return ''
        statements = WIKIDATA_STATEMENTS
        random.shuffle(statements)

        for sta in statements:
            if not sta:
                instances = entity.getlist(instance)
                human = client.get(EntityId(WIKIDATA_INSTANCE['human']))
                if human in instances:
                    head_phrase = CLAUSE_HEAD[0].replace('which', 'who')
                    return head_phrase + str(entity.description)
                else:
                    return CLAUSE_HEAD[0] + str(entity.description)
            else:
                statement_entity = client.get(sta)
                statement = entity.getlist(statement_entity)
                if statement:
                    for state in statement:
                        if isinstance(state, Entity):
                            return CLAUSE_HEAD[sta] + str(state.label)

    def get_clause(self, query):
        r"""
        Generate a subordinate clause for the given query.

        :param str query: A list of words in the entity,
            which is joined by '%20'.
        :return: The subordinate clause generated.

        """
        searched_list = self.search_list(query)
        for entity_info in searched_list:
            clause_add = self.clause_generate(entity_info['id'])
            if clause_add:
                return clause_add

    def pop_list(self, entity_list, pop_set):
        for i in sorted(list(pop_set), reverse=True):
            entity_list.pop(i)
        return None

    def find_entity(self, sample):
        r"""
        Find an entity in the sentence.

        :param ~DPSample sample:
        :return: A list of entities, long to short.

        """
        brackets = sample.brackets
        words = sample.get_words('x')
        postags = sample.get_value('postag')

        nnp_list = []
        for i, postag in enumerate(postags):
            if postag in ('NNP', 'NNPS'):
                nnp_list.append(i + 1)

        entity_list = []
        if nnp_list:
            entity_list.append([nnp_list[0]])
            nnp_list.pop(0)
            for word in nnp_list:
                if word == entity_list[-1][-1] + 1:
                    entity_list[-1].append(word)
                else:
                    entity_list.append([word])
            self.exclude_inside_brackets(entity_list, brackets)
            self.exclude_followed(entity_list, words, postags)
            self.combine_entities(entity_list, words)

        return sorted(entity_list, key=len, reverse=True)

    def exclude_inside_brackets(self, entity_list, brackets):
        pop_set = set()
        if brackets:
            for i, entity in enumerate(entity_list):
                for pair in brackets:
                    if pair[0] < entity[0] < pair[1]:
                        pop_set.add(i)
        self.pop_list(entity_list, pop_set)

    def exclude_followed(self, entity_list, words, postags):
        pop_set = set()
        for i, entity in enumerate(entity_list):
            if postags[entity[-1]] in ('POS', '-LRB-'):
                pop_set.add(i)
            if words[entity[-1]] == ',':
                if postags[entity[-1] + 1] in ('WP$', 'WP', 'WDT', 'WRB'):
                    pop_set.add(i)
        self.pop_list(entity_list, pop_set)

    def combine_entities(self, entity_list, words):
        pop_set = set()
        for i, entity in enumerate(entity_list):
            if i < len(entity_list) - 1:
                if entity[-1] + 2 == entity_list[i + 1][0]:
                    if words[entity[-1]] in ('and', '&'):
                        entity.append(entity[-1] + 1)
                        entity.extend(entity_list[i + 1])
                        pop_set.add(i + 1)
        self.pop_list(entity_list, pop_set)


