import numpy as np


def _get_entailed_hypothesis(tokens, index, number):
    """
    return a sentence that have the entailment semantic with the original sample
    :param tokens: the text
    :param index: the position for the number word
    :param number: number word (int)
    :return: The sentence changed the number word
    """

    number = str(number)
    new_digit = np.random.randint(1, 9)
    old_digit = int(number[0])
    while new_digit == old_digit:
        new_digit = np.random.randint(1, 9)
    new_num = str(new_digit) + number[1:]
    new_tokens = []
    if old_digit < new_digit:
        new_tokens = tokens[:index] + \
            ['less than', new_num] + tokens[index + 1:]
    else:
        new_tokens = tokens[:index] + \
            ['more than', new_num] + tokens[index + 1:]
    return ' '.join(new_tokens)


def _get_contradictory_hypothesis(tokens, index, number):
    """
    return a sentence that have the contradicition semantic with the original
    sample
    :param tokens: the text
    :param index: the position for the number word
    :param number: number word (int)
    :return: The sentence changed the number word

    """
    prob = np.random.binomial(1, 0.5)

    if prob < 0.5:
        number = str(number)
        new_digit = np.random.randint(1, 9)
        old_digit = int(number[0])
        while new_digit == old_digit:
            new_digit = np.random.randint(1, 9)
        new_num = str(new_digit) + number[1:]
        new_tokens = tokens[:index] + [new_num] + tokens[index + 1:]
    else:
        prob2 = np.random.binomial(1, 0.5)
        if prob2 < 0.5:
            new_tokens = tokens[:index] + \
                ['more than', str(number)] + tokens[index + 1:]
        else:
            new_tokens = tokens[:index] + \
                ['less than', str(number)] + tokens[index + 1:]

    return ' '.join(new_tokens)
