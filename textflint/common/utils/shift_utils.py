def shift_maker(sign_idx, shf):
    """ 
    Makes `shift`, which is a basic shift function to be composed.
        `shift`: if idx >= sign_idx, right shift shf for the idx.
    :param int sign_idx: word after this idx should shift
    :param int shf: word shift
    """
    return lambda idx: 0 if idx < sign_idx else shf


def shift_collector(shifts):
    """ 
    Collect and compose `shift`s to a general `shift`, to be applied to
        each span (or sth else).
    :param list shifts: the shift functions
    :return ~types.FunctionType: the collected shift function
    """
    return lambda idx: idx + sum(map(lambda f: f(idx), shifts))


def shift_decor(shift_func):
    """ 
    Make `shift` error-free on non-int values. Decorated `shift` keeps non-int 
        values original.
    :param ~types.FunctionType: a shift function that only processes int values
    :return ~types.FunctionType: a shift function that processes all types of values
    """
    def wrapped(maybe_word_idx):
        if isinstance(maybe_word_idx, int):
            return shift_func(maybe_word_idx)
        else:
            return maybe_word_idx
    return wrapped