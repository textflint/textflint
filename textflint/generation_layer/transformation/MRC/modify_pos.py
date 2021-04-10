r"""
Modify position of the sentence that contains answer
==========================================================
"""
__all__ = ['ModifyPos']

from ..transformation import Transformation


class ModifyPos(Transformation):
    r"""
    Modify position of the sentence that contains the answer.
    """

    def __repr__(self):
        return 'ModifyPosition'

    def _transform(self, sample, **kwargs):
        r"""
        Extract the trivial sentence without answers and insert it
            at the beginning or the end.
        :param sample: the sample to transform
        :param kwargs:
        :return: list of sample
        """
        # filter no-answer samples
        if sample.is_impossible:
            return []
        answers = sample.get_answers()
        sents = sample.get_sentences('context')
        sent_start = 0
        original_idx = -1
        trivial_sent = None
        for idx, sent in enumerate(sents):
            trivial = True
            sent_len = len(self.processor.tokenize(sent))
            for answer in answers:
                if answer['start'] >= sent_start and answer['end'] \
                        <= sent_start + sent_len:
                    trivial = False
                    break
            if trivial:
                sent_span = [sent_start, sent_start + sent_len]
                trivial_sent = sent
                original_idx = idx
                break
        if trivial_sent == " ":
            return []
        if trivial_sent is None:
            return []
        # Insert the sentence at the begin or at the end
        sample = sample.delete_field_at_indices('context', [sent_span])
        if original_idx == 0:
            trans_sample = sample.insert_field_after_index(
                'context', len(sample.get_words('context')) - 1, trivial_sent)
        else:
            trans_sample = sample.insert_field_before_index(
                'context', 0, trivial_sent)
        return [trans_sample]



