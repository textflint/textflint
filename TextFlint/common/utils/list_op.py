import copy
import random
from .logger import logger


def trade_off_sub_words(sub_words, sub_indices, trans_num=None, n=1):
    r"""
    Select proper candidate words to maximum number of transform result.
    Select words of top n substitutes words number.

    :param list sub_words: list of substitutes word of each legal word
    :param list sub_indices: list of indices of each legal word
    :param int trans_num: max number of words to apply substitution
    :param int n:
    :return: sub_words after alignment + indices of sub_words
    """

    assert len(sub_words) == len(sub_indices), \
        "The length of sub_words and sub_indices should be the same"
    # max number of words to apply transform
    trans_num = min(trans_num, len(sub_words)) if trans_num else len(sub_words)

    re_sub_words = []
    re_sub_indices = []
    for i, j in zip(sub_words, sub_indices):
        if isinstance(i, list) and i:
            re_sub_words.append(i)
            re_sub_indices.append(j)

    if trans_num < len(re_sub_words):
        trans_info = [[i, j] for i, j in zip(re_sub_words, re_sub_indices)]
        trans_info = random.sample(trans_info, trans_num)
        re_sub_words = [i for i, j in trans_info]
        re_sub_indices = [j for i, j in trans_info]

    if not re_sub_words:
        return [], []

    return descartes(re_sub_words, n), re_sub_indices


def normalize_scope(scope):
    r"""
    Convert various scope input to list format of [left_bound, right_bound]

    :param int|list|tuple|slice scope:
        can be int indicate replace single item like 1 or 3.
        can be list like (0,3) indicate replace items from 0 to 3(not included)
        or their list like [5,6]
        can be slice which would be convert to list or their list.
    :return list: [left_bound, right_bound]
    """

    if not isinstance(scope, (list, tuple, int, slice)):
        raise TypeError(
            f"_replace_at_scopes requires list of ``list``, "
            f"``tuple`` , ``int`` or ``slice``, got {type(scope)}"
        )

    # convert int to list
    if isinstance(scope, int):
        if scope < 0:
            raise ValueError(
                f"Can't replace {scope} index"
            )
        scope = [scope, scope + 1]

    # convert slice to list, default step of slice is 1
    if isinstance(scope, slice):
        scope = [scope.start, scope.stop]

    if not all(isinstance(bound, int) for bound in scope):
        raise TypeError(
            f"normalize_scope requires ``int`` elements, got {scope}"
        )

    if len(scope) > 2:
        raise ValueError(
            f"normalize_scope requires range not longer than 2, "
            f"got {len(scope)}"
        )
    elif len(scope) == 2 and scope[0] > scope[1]:
        raise ValueError(
            f"normalize_scope requires left bound not bigger than right bound, "
            f"got {scope}"
        )

    return list(scope)


def _replace_at_scopes(origin_list, scopes, new_items):
    r"""
    Replace items by ranges.

    :param list origin_list: the list value to be modified.
    :param list scopes: list of int/list/tuple/slice
        can be int indicate replace single item or their list like [1, 2, 3].
        can be list like (0,3) indicate replace items from 0 to 3(not included)
            or their list like [(0, 3), (5,6)]
        can be slice which would be convert to list or their list.
    Watch out! Each range must be the same type!
    :param list new_items: items corresponding scopes.
    :return list: new list
    """

    items = copy.deepcopy(origin_list)
    scopes = copy.deepcopy(scopes)
    assert isinstance(scopes, list), f"The type of scopes should be `list`, " \
                                     f"not {type(scopes)}."

    if new_items is None:
        raise ValueError(
            f"Invalid replace input : {new_items}."
        )

    if len(new_items) != len(scopes):
        raise ValueError(
            f"Cannot replace {len(new_items)} tokens at {len(scopes)} ranges."
        )

    # ranges check
    for idx, scope_i in enumerate(scopes):
        scope_i = list(normalize_scope(scope_i))
        scope_i[0] = max(0, min(scope_i[0], len(items)))
        scope_i[1] = min(len(items), scope_i[1])
        scopes[idx] = scope_i

        if scope_i[0] >= scope_i[1]:
            raise ValueError(
                f"No elements selected in range between "
                f"{scope_i[0]} and {scope_i[1]}"
            )

    # check whether exist range collision
    def check_collision(r):
        for i, range1 in enumerate(r):
            for j, range2 in enumerate(r[i + 1:]):
                l1, r1 = range1
                l2, r2 = range2
                if max(l1, l2) < min(r1, r2):
                    return True
        return False

    if check_collision(scopes):
        raise ValueError(
            f"Ranges {scopes} has collision"
        )

    real_items = []

    for idx, item in enumerate(new_items):
        next_item = item
        if not isinstance(item, (list, tuple)):
            next_item = [item]
        if item in [
            None,
            '',
                []]:  # Assume token is empty if it's ``None``, ``[]``, ``''``
            next_item = []
        real_items.append(list(next_item))

    sorted_items, sorted_scopes = zip(
        *sorted(zip(real_items, scopes), key=lambda x: x[1]))
    sorted_scopes = list(sorted_scopes)
    sorted_scopes.append([len(items), -1])
    replaced_items = items[:sorted_scopes[0][0]]

    for idx, sorted_token in enumerate(sorted_items):
        replaced_items.extend(sorted_token)
        replaced_items.extend(
            items[sorted_scopes[idx][1]: sorted_scopes[idx + 1][0]])

    return replaced_items


def replace_at_scopes(origin_list, scopes, new_items):
    r"""
    Replace items of given list.
    Notice: just support isometric replace.

    :param list origin_list:
    :param list scopes: list of int/list/slice
        can be int indicate replace single item or their list like [1, 2, 3].
        can be list like (0,3) indicate replace items from 0 to 3(not included)
            or their list like [(0, 3), (5,6)]
        can be slice which would be convert to list or their list.
        Watch out! Each range must be the same type!
    :param list new_items: items corresponding scopes.
    :return:
    """

    scopes = copy.deepcopy(scopes)
    assert isinstance(origin_list, list), \
        f"Origin_list should be `list`, not `{type(origin_list)}`."
    assert isinstance(scopes, list), \
        f"Scopes should be `list`, not `{type(scopes)}`."
    assert isinstance(new_items, list), \
        f"New_items should be `list`, not `{type(new_items)}`."
    assert len(scopes) == len(new_items), \
        f"The length of scopes `{len(scopes)}` and the length of new_items " \
        f"`{len(new_items)}` should be the same."

    # check whether scope and items length is compatible
    for i in range(len(scopes)):
        scopes[i] = scope = normalize_scope(scopes[i])
        items = new_items[i] if isinstance(
            new_items[i], (list, tuple)) else [new_items[i]]
        assert scope[1] - scope[0] == len(items), \
            f"scope is not compatible with items length `{len(items)}`."

    return _replace_at_scopes(origin_list, scopes, new_items)


def replace_at_scope(origin_list, scope, new_items):
    r"""
    Replace items of given list instance.

    :param list origin_list:
    :param int|list|slice scope:
        can be int indicate replace single item or their list like 1.
        can be list like (0,3) indicate replace items from 0 to 3(not included)
            or their list like [0, 3]
        can be slice which would be convert to list or their list.
    :param new_items: list
    :return list: new list
    """

    return replace_at_scopes(origin_list, [scope], [new_items])


def delete_at_scopes(origin_list, scopes):
    r"""
    Delete items of origin_list of given scopes.

    :param list origin_list:
    :param list scopes: list of int/list/tuple/slice
        can be int indicate replace single item or their list like [1, 2, 3].
        can be list like (0,3) indicate replace items from 0 to 3(not included)
            or their list like [5,6]
        can be slice which would be convert to list or their list.
    :return list: new list
    """

    scopes = copy.deepcopy(scopes)
    assert isinstance(scopes, list), f"Scopes should be `list`, not `{type(scopes)}`."

    # check whether scope and items length is compatible
    for i in range(len(scopes)):
        scopes[i] = normalize_scope(scopes[i])

    return _replace_at_scopes(origin_list, scopes, [None] * len(scopes))


def delete_at_scope(origin_list, scope):
    r"""
    Delete items of origin_list of given scope.

    :param list origin_list:
    :param int|list|tuple|slice scope:
        can be int indicate replace single item or their list like [1, 2, 3].
        can be list like (0,3) indicate replace items from 0 to 3(not included)
            or their list like [5,6]
        can be slice which would be convert to list or their list.
    :return:
    """

    return delete_at_scopes(origin_list, [scope])


def handle_empty_insertion(new_items):
    r"""
    Handle inserting new items to an empty list, by concatenating all new items
    Warning if multiple items are fed.

    :param list new_items: list
    :return list: new list
    """

    if len(new_items) > 1:
        logger.warning("Trying to add multiple items into an empty list.")
    appended_items = []
    for items in new_items:
        appended_items.extend([items] if not isinstance(items, list) else items)
    return appended_items


def insert_before_indices(origin_list, indices, new_items):
    r"""
    Insert items to origin_list before given indices.

    :param list origin_list:
    :param list indices:
    :param list new_items:
    :return list:
    """

    assert isinstance(indices, list), \
        f"Indices should be `list`, not `{type(list)}`."
    assert isinstance(new_items, list), \
        f"New_items should be `list`, not `{type(new_items)}`"
    assert len(indices) == len(new_items), \
        f"Indices length `{len(indices)}` and items length `{len(new_items)}'" \
        f" should be the same."
    new_items = copy.deepcopy(new_items)

    if len(origin_list) == 0:
        return handle_empty_insertion(new_items)

    insert_items = []
    for index, new_item in enumerate(new_items):
        if not isinstance(new_item, list):
            new_item = [new_item]
        items = copy.deepcopy(new_item)
        items.extend([origin_list[indices[index]]])
        insert_items.append(items)

    return _replace_at_scopes(origin_list, indices, insert_items)


def insert_before_index(origin_list, index, new_items):
    r"""
    Insert items to origin_list before given index.

    :param list origin_list:
    :param int index:
    :param list new_items:
    :return list:
    """
    return insert_before_indices(origin_list, [index], [new_items])


def insert_after_indices(origin_list, indices, new_items):
    r"""
    Insert items to origin_list after given indices.

    :param list origin_list:
    :param list indices:
    :param list new_items:
    :return list:
    """

    assert isinstance(indices, list), \
        f"Indices should be `list`, not `{type(indices)}`."
    assert isinstance(new_items, list), \
        f"New_items should be `list`, not `{type(new_items)}`."
    assert len(indices) == len(new_items), \
        f"Indices length `{len(indices)}` and items length `{len(new_items)}'" \
        f" should be the same."
    new_items = copy.deepcopy(new_items)

    if len(origin_list) == 0:
        return handle_empty_insertion(new_items)

    insert_items = []
    for index, new_item in enumerate(new_items):
        if not isinstance(new_item, list):
            new_item = [new_item]
        items = [origin_list[indices[index]]]
        items.extend(new_item)
        insert_items.append(items)

    return _replace_at_scopes(origin_list, indices, insert_items)


def insert_after_index(origin_list, index, new_items):
    r"""
    Insert items to origin_list after given index.

    :param list origin_list:
    :param int index:
    :param list new_items:
    :return:
    """

    return insert_after_indices(origin_list, [index], [new_items])


def swap_at_index(origin_list, first_index, second_index):
    r"""
    Swap items between first_index and second_index of origin_list

    :param list origin_list:
    :param int first_index:
    :param int second_index:
    :return list:
    """

    if max(first_index, second_index) > len(origin_list) - 1:
        raise ValueError(
            f"Can't swap {0} and {1} index to items {2} of {3} length" .format(
                first_index,
                second_index,
                origin_list,
                len(origin_list)))

    return _replace_at_scopes(
        origin_list, [
            first_index, second_index], [
            origin_list[second_index], origin_list[first_index]])


def unequal_replace_at_scopes(origin_list, scopes, new_items):
    r"""
    Replace items of given list.

    Notice: support unequal replace.
    :param list origin_list:
    :param list scopes: list of int/list/slice
        can be int indicate replace single item or their list like [1, 2, 3].
        can be list like (0,3) indicate replace items from 0 to 3(not included)
            or their list like [(0, 3), (5,6)]
        can be slice which would be convert to list or their list.
    :param new_items:
    :return list :
    """

    scopes = copy.deepcopy(scopes)
    assert isinstance(scopes, list), \
        f"Scopes should be `list`, not `{type(scopes)}`."
    assert isinstance(new_items, list), \
        f"New_items should be `list`, not `{type(new_items)}"
    assert len(scopes) == len(new_items), \
        f"Scopes length `{len(scopes)}` and items length `{len(new_items)}' " \
        f"should be the same."

    return _replace_at_scopes(origin_list, scopes, new_items)


def get_align_seq(align_items, value):
    r"""
    Get values which shape align with align items.

    :param list align_items:
    :param str value:
    :return list: list which align with align_items.
    """

    if isinstance(align_items[0], list):
        new_seq = [[value] * len(i) for i in align_items]
    else:
        new_seq = len(align_items) * [value]

    return new_seq


def descartes(calculation_items, n):
    """
    :param list calculation_items:
    :param int n: quantity to select
    :return list: list items which we random choice from Cartesian product.
    """

    assert isinstance(calculation_items, list), \
        f"Calculation_items should be list, not `{type(calculation_items)}`."
    assert isinstance(n, int) and n > 0, \
        f"n should be int and n should be greater than 0."
    if not calculation_items:
        raise ValueError(f"An empty vector cannot participate in operation")
    if len(calculation_items) == 1:
        res = _descartes_check(calculation_items[0], n)
        return [[k] for k in res]
    choice_descartes = [[item] for item in calculation_items[0]]
    for i in range(1, len(calculation_items)):
        choice_descartes = _get_descartes(
            choice_descartes, calculation_items[i], n)
    return choice_descartes


def _get_descartes(vector_a, vector_b, n):
    """
    Get the Cartesian product of two vectors and choice n of result.

    :param list vector_a: list [[vector1], [vector2],......]
    :param list vector_b: list [x * items]
    :param int n: quantity to select int
    :return list: list items which we random choice from Cartesian product.
    """

    assert isinstance(n, int), f"n should be `int`, not `{type(n)}`."
    vector_a = _descartes_check(vector_a, n)
    vector_b = _descartes_check(vector_b, n)
    descartes = []
    for a in vector_a:
        for b in vector_b:
            descartes.append(a + [b])
    if len(descartes) > n:
        return random.sample(descartes, n)
    return descartes


def _descartes_check(vector_a, n):
    """
    Check the vector and choice n from it.

    :param list vector_a:
    :param int n: quantity to select
    :return list: list items which we random choice from vector.
    """

    assert isinstance(vector_a, list), \
        f"Vector_a should be `list`, not `{type(vector_a)}`."
    if not vector_a:
        raise ValueError(f"An empty vector cannot compute a Cartesian product")

    vector = []
    for v in vector_a:
        if v not in vector:
            vector.append(v)

    if len(vector) > n:
        return random.sample(vector, n)

    return vector
