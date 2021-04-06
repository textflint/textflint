import random
import string


def get_start_end(word, skip_first=False, skip_last=False):
    """
    Get valid operation range of one word.

    :param str word: target word string
    :param bool skip_first: whether operate first char
    :param bool skip_last: whether operate last char
    :return: start index, last index
    """

    chars = list(word)
    start = int(skip_first)
    end = len(chars) - 1 - int(skip_last)

    return start, end


def get_random_letter(src_char=None):
    """
    Get replaced letter according src_char format.

    :param char src_char:
    :return: default return a lower letter
    """

    if src_char.isdigit():
        return random.choice(string.digits)
    if src_char.isupper():
        return random.choice(string.ascii_uppercase)
    else:
        return random.choice(string.ascii_lowercase)


def swap(word, num=1, skip_first=False, skip_last=False):
    """
    Swaps random characters with their neighbors.

    :param str word: target word
    :param int num: number of typos to add
    :param bool skip_first: whether swap first char of word
    :param bool skip_last: whether swap last char of word
    :return: perturbed strings
    """

    if len(word) <= 1:
        return word

    chars = list(word)
    start, end = get_start_end(word, skip_first, skip_last)

    # error swap num, return original word
    if end - start < num:
        return None

    swap_idxes = random.sample(list(range(start, end)), num)

    for swap in swap_idxes:
        tmp = chars[swap]
        chars[swap] = chars[swap + 1]
        chars[swap + 1] = tmp

    return ''.join(chars)


def insert(word, num=1, skip_first=False, skip_last=False):
    """
    Perturb the word with 1 random character inserted.

    :param str word: target word
    :param int num: number of typos to add
    :param bool skip_first: whether insert char at the beginning of word
    :param bool skip_last: whether insert char at the end of word
    :return: perturbed strings
    """

    if len(word) <= 1:
        return word

    chars = list(word)
    start, end = get_start_end(word, skip_first, skip_last)

    if end - start + 2 < num:
        return None

    swap_idxes = random.sample(list(range(start, end + 2)), num)
    swap_idxes.sort(reverse=True)

    for idx in swap_idxes:
        insert_char = get_random_letter(chars[min(idx, len(chars) - 1)])
        chars = chars[:idx] + [insert_char] + chars[idx:]

    return "".join(chars)


def delete(word, num=1, skip_first=False, skip_last=False):
    """
    Perturb the word wityh 1 letter deleted.

    :param str word: number of typos to add
    :param int num: number of typos to add
    :param bool skip_first: whether delete the char at the beginning of word
    :param bool skip_last: whether delete the char at the end of word
    :return: perturbed strings
    """

    if len(word) <= 1:
        return word

    chars = list(word)
    start, end = get_start_end(word, skip_first, skip_last)

    if end - start + 1 < num:
        return None

    swap_idxes = random.sample(list(range(start, end + 1)), num)
    swap_idxes.sort(reverse=True)

    for idx in swap_idxes:
        chars = chars[:idx] + chars[idx + 1:]

    return "".join(chars)


def replace(word, num=1, skip_first=False, skip_last=False):
    """
    Perturb the word with 1 letter substituted for a random letter.

    :param str word: target word
    :param int num: number of typos to add
    :param bool skip_first: whether replace the char at the beginning of word
    :param bool skip_last: whether replace the char at the beginning of word
    :return: perturbed strings
    """

    if len(word) <= 1:
        return []

    chars = list(word)
    start, end = get_start_end(word, skip_first, skip_last)

    # error swap num, return original word
    if end - start + 1 < num:
        return word

    idxes = random.sample(list(range(start, end + 1)), num)

    for idx in idxes:
        chars[idx] = get_random_letter(chars[idx])

    return "".join(chars)
