r"""
Generator for ABSA Task
============================================

"""
__all__ = ['ABSAGenerator']

from tqdm import tqdm
from .generator import Generator
from ...common.utils.logger import logger
from ...input.component.sample import ABSASample
from ..transformation.transformation import Transformation
from ...common.utils.load import absa_dict_loader
from ...common.utils.install import download_if_needed
from ...common.settings import TASK_TRANSFORMATION_PATH, \
    ALLOWED_TRANSFORMATIONS, TASK_SUBPOPULATION_PATH, \
    ALLOWED_SUBPOPULATIONS, ABSA_TRAIN_RESTAURANT_PATH, \
    ABSA_TRAIN_LAPTOP_PATH

Flint = {
    "transformation": {
        'task_path': TASK_TRANSFORMATION_PATH,
        'allowed_methods': ALLOWED_TRANSFORMATIONS
    },
    "subpopulation": {
        'task_path': TASK_SUBPOPULATION_PATH,
        'allowed_methods': ALLOWED_SUBPOPULATIONS
    }
}


class ABSAGenerator(Generator):
    r"""
    Generate extra text for AbsaAddDiff,
    and dataset type is assigned in configure.

    """

    def __init__(
        self,
        task='ABSA',
        fields='sentence',
        max_trans=1,
        trans_methods=None,
        trans_config=None,
        return_unk=True,
        sub_methods=None,
        sub_config=None,
        attack_methods=None,
        validate_methods=None,
        dataset_config='restaurant',
        **kwargs
    ):
        self.dataset_config = dataset_config
        self.nlp = Transformation.processor.nlp
        self.extra_text = []
        super().__init__(
            task=task,
            max_trans=max_trans,
            fields=fields,
            trans_methods=trans_methods,
            trans_config=trans_config,
            return_unk=return_unk,
            sub_methods=sub_methods,
            sub_config=sub_config,
            attack_methods=attack_methods,
            validate_methods=validate_methods,
            **kwargs
        )
        self.transform_methods = trans_methods
        if self.dataset_config is None:
            logger.info(
                '******No config of dataset is available for AddDiff!******'
            )
            if 'AddDiff' in ALLOWED_TRANSFORMATIONS['ABSA']:
                ALLOWED_TRANSFORMATIONS['ABSA'].remove('AddDiff')
        else:
            if 'AddDiff' in trans_methods and \
                    self.dataset_config == 'restaurant':
                self.examples = absa_dict_loader(
                    download_if_needed(ABSA_TRAIN_RESTAURANT_PATH))
                self.extra_text = self.get_extra_text()
            else:
                for transform_method in trans_methods:
                    if 'AddDiff' in transform_method and \
                            self.dataset_config == 'laptop':
                        self.examples = absa_dict_loader(
                            download_if_needed(ABSA_TRAIN_LAPTOP_PATH))
                        self.extra_text = self.get_extra_text()
                        break

    @staticmethod
    def get_extra_sentence(term_list, term_id, phrases):
        r"""
        Get the extra sentence from phrases text.

        :param dict term_list: term list
        :param str term_id: term id
        :param list phrases: phrase list
        :return list: extra sentences
        """
        phrases_list = []
        other_terms = []
        extra_sentence = []
        term = term_list[term_id]['term']
        opinions = term_list[term_id]['opinion_words']
        for other_id in term_list:
            if other_id != term_id:
                other_terms.append(term_list[other_id]['term'])
        for phrase in phrases:
            opinion_exist = True
            phrase_ = \
                ''.join([token.text_with_ws for token in list(phrase)]).strip()
            for opinion_word in opinions:
                if opinion_word not in phrase_:
                    opinion_exist = False
            if term in phrase_ and opinion_exist is True:
                phrases_list.append(phrase_)
        for phrase in phrases_list:
            overlap = False
            for other_term in other_terms:
                if other_term in phrase:
                    overlap = True
                    break
            if not overlap:
                extra_sentence.append(phrase)
        extra_sentence = sorted(extra_sentence, key=len)
        return extra_sentence

    def get_extra_text(self):
        r"""
        Get extra text from training dataset.

        :return: dict of extra text
        """
        positive_text = []
        negative_text = []
        neutral_text = []
        logger.info('******Prepare extra {0} corpus for AddDiff!******'
                    .format(self.dataset_config))
        for text_id in tqdm(self.examples):
            text = self.examples[text_id]
            text_sample = ABSASample(text)
            term_list = text_sample.term_list
            text_doc = self.nlp(text_sample.sentence.text)
            phrase_list = []
            for token in text_doc:
                if len(list(token.subtree)) > 1:
                    phrase_list.append(list(token.subtree))
            for term_id in term_list:
                term = term_list[term_id]['term']
                term_polarity = term_list[term_id]['polarity']
                extra_sentence = \
                    self.get_extra_sentence(term_list, term_id, phrase_list)
                if len(extra_sentence) == 0:
                    continue
                extra_sentence = extra_sentence[0]

                if term_polarity == 'positive':
                    positive_text.append((term.lower(),
                                          [extra_sentence.lower()]))
                elif term_polarity == 'negative':
                    negative_text.append((term.lower(),
                                          [extra_sentence.lower()]))
                elif term_polarity == 'neutral':
                    neutral_text.append((term.lower(),
                                         [extra_sentence.lower()]))

        return {'positive': positive_text,
                'negative': negative_text,
                'neutral': neutral_text}

    def generate_by_transformations(self, dataset, **kwargs):
        r"""
        Generate samples by a list of transformation methods.

        :param dataset: the input dataset
        :return: (original samples, new samples, generated function string)
        """
        self.prepare(dataset)

        for trans_obj in self._get_flint_objs(
            self.transform_methods,
            TASK_TRANSFORMATION_PATH,
            ALLOWED_TRANSFORMATIONS
        ):
            # initialize current index of dataset
            dataset.init_iter()

            logger.info('******Start {0}!******'.format(trans_obj))
            generated_samples = dataset.new_dataset()
            original_samples = dataset.new_dataset()

            for sample in tqdm(dataset):
                # default return list of samples
                trans_rst = trans_obj.transform(
                    sample,
                    n=self.max_trans,
                    field=self.fields,
                    extra_text=self.extra_text)
                if trans_rst:
                    generated_samples.extend(trans_rst)
                    original_samples.append(sample)

            yield original_samples, generated_samples, trans_obj.__repr__()
            logger.info('******Finish {0}!******'.format(trans_obj))
