"""
NLTK tokenize and its reverse tokenize function
============================================

"""

import re
import spacy
from functools import reduce

from ...common.utils.install import download_if_needed
from ...common.settings import MODEL_PATH_WEB, MODEL_PATH

nlp = spacy.load(download_if_needed(MODEL_PATH_WEB) + MODEL_PATH)


def sentence_tokenize(text):
    assert isinstance(text, str)
    text = nlp.tokenizer(text)
    for pipeline in nlp.pipeline:
        if 'sentencizer' in pipeline:
            return [sent.text for sent in pipeline[1](text).sents]
    nlp.add_pipe('sentencizer')

    return [sent.text for sent in nlp.pipeline[-1][1](text).sents]


def tokenize_one_sent(text, split_by_space=False):
    """ Split one sentence.

    Args:
        text: string
        split_by_space: bool

    Returns:
        list of tokens

    """
    assert isinstance(text, str)
    if split_by_space:
        return text.split(" ")
    else:
        return [
            word.text.replace("''", '"')
                .replace("``", '"') for word in nlp.tokenizer(text) if word.text != ' ' * len(word)]


def tokenize(text, is_one_sent=False, split_by_space=False):
    """ Split a text into tokens (words, morphemes we can separate such as
        "n't", and punctuation).

    Args:
        text: string
        is_one_sent: bool
        split_by_space: bool

    Returns:
        list of tokens

    """
    assert isinstance(text, str)
    def _tokenize_gen(text):
        if is_one_sent:
            yield tokenize_one_sent(text, split_by_space=split_by_space)
        else:
            for sent in sentence_tokenize(text):
                yield tokenize_one_sent(sent, split_by_space=split_by_space)

    return reduce(lambda x, y: x + y, list(_tokenize_gen(text)), [])


def untokenize(words):
    """ Untokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `untokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.

        Watch out!
        Default punctuation add to the word before its index, it may raise inconsistency bug.

    Args:
        words: list

    Returns:
        sentence string.

    """
    assert isinstance(words, list)
    text = ' '.join(words)
    step1 = text.replace("`` ", '"')\
        .replace(" ''", '"')\
        .replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
        "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    step7 = step6.replace('do nt', 'dont').replace('Do nt', 'Dont')
    step8 = step7.replace(' - ', '-')
    return step8.strip()
