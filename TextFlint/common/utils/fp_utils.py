def concat(xss):
    r""" 
    Concat list of list to be a list.
    Usage::

        concat([[1, 2], [2, 3]]) == [1, 2, 2, 3]

    :param list xss: the list to be concat
    """
    from functools import reduce
    if len(xss) == 0:
        return []
    return reduce(lambda x, y: x+y, xss)


def recur_ap(f, ls):
    """ 
    Apply `f` to every elem in `ls` (a nested list) recursively. Usages::

        recur_ap(lambda x: x+2, 1) = 3
        recur_ap(lambda x: x+2, [2, [3, 4]]) = [4, [5, 6]]

    :param ~types.FunctionType f: the function to be applied to ls
    :param ls: the value or the nested list to be processed
    :return: process result
    """
    if isinstance(ls, list):
        return [recur_ap(f, elem) for elem in ls]
    else:
        return f(ls)
