import os
from pathlib import Path
import platform
import copy

# default settings
current_path = Path(__file__).resolve().parent
if platform.system() == 'Windows':
    CACHE_DIR = os.environ.get(
        "TR_CACHE_DIR", os.path.join(os.environ.get('APPDATA'), 'textflint')
    )
else:
    CACHE_DIR = os.environ.get(
        "TR_CACHE_DIR", os.path.expanduser("~/.cache/textflint")
    )
os.environ["TR_CACHE_DIR"] = CACHE_DIR
DATA_PATH = os.path.join(current_path, './res/')
GENERATOR_PATH = os.path.join(current_path, './../generation/generator/')
SAMPLE_PATH = os.path.join(current_path, './../input/component/sample/')
TRANSFORMATION_PATH = os.path.join(current_path,
                                   './../generation/transformation/')
SUBPOPULATION_PATH = os.path.join(current_path,
                                  './../generation/subpopulation/')

# mask settings
ORIGIN = 0
TASK_MASK = 1
MODIFIED_MASK = 2

NLP_TASK_MAP = {
    'UT': 'Universal transform',
    'ABSA': 'Aspect Based Sentiment Analysis',
    'SA': 'Sentiment Analysis',
    'CWS': 'Chinese Word Segmentation',
    'NER': 'Named Entity Recognition',
    'POS': 'Part-of-Speech Tagging',
    'DP': 'Dependency Parsing',
    'MRC': 'Machine Reading Comprehension',
    'SM': 'Semantic Matching',
    'NLI': 'Natural language inference',
    'RE': 'Relation Extraction',
    'COREF': 'Coreference resolution'
}

TRANSFORM_FIELDS = {
    'UT': 'x',
    'ABSA': 'sentence',
    'SA': 'x',
    'CWS': 'x',
    'NER': 'text',
    'POS': 'x',
    'DP': 'x',
    'MRC': 'context',
    'SM': ['sentence1', 'sentence2'],
    'NLI': ['premise', 'hypothesis'],
    'RE': 'x',
    'COREF': 'x'
}
TASK_SUBPOPULATION_PATH = dict(
    (task, os.path.join(SUBPOPULATION_PATH, task))
    for task in NLP_TASK_MAP
)

TASK_TRANSFORMATION_PATH = dict(
    (task, os.path.join(TRANSFORMATION_PATH, task))
    for task in NLP_TASK_MAP
)

# indicate allowed subpopulations of specific task
UT_SUBPOPULATIONS = [
    "LengthSubPopulation",
    "LMSubPopulation",
    "PhraseSubPopulation",
    "PrejudiceSubPopulation"
]
ALLOWED_SUBPOPULATIONS = {
    key: copy.copy(UT_SUBPOPULATIONS) for key in NLP_TASK_MAP if key != 'CWS'
}
ALLOWED_SUBPOPULATIONS['CWS'] = []

UT_TRANSFORMATIONS = [
    'InsertAdv',
    'AppendIrr',
    'BackTrans',
    'WordCase',
    'Contraction',
    'SwapNamedEnt',
    'Keyboard',
    'MLMSuggestion',
    'SwapNum',
    'Ocr',
    'Punctuation',
    'ReverseNeg',
    'SpellingError',
    'Tense',
    'TwitterType',
    'Typos',
    'SwapSynWordEmbedding',
    'SwapAntWordNet',
    'SwapSynWordNet',
    'Prejudice'
]

UNMATCH_UT_TRANSFORMATIONS = {
    'UT': [],
    'ABSA': [
        'ReverseNeg',
        'BackTrans',
        'SwapAntWordNet'
    ],
    'SA': [
        'ReverseNeg',
        'SwapAntWordNet'
    ],
    'CWS': copy.copy(UT_TRANSFORMATIONS),
    'NER': [
        'BackTrans',
        'SwapNamedEnt',
        'Prejudice'
    ],
    'POS': [
        'BackTrans',
        'Tense'
    ],
    'DP': [
        'BackTrans'
    ],
    'MRC': [
        'SwapNum',
        'SwapAntWordNet',
        'ReverseNeg',
        'SwapNamedEnt'
        ],
    'NLI': [],
    'SM': [],
    'COREF': [
        'BackTrans'
    ],
    'RE': []
}

TASK_TRANSFORMATIONS = {
    'UT': UT_TRANSFORMATIONS,
    'ABSA': [
        'RevTgt',
        'RevNon',
        'AddDiff',
    ],
    'SA': [
        'SwapSpecialEnt',
        'AddSum',
        'DoubleDenial'
    ],
    'CWS': [
        'SwapName',
        'CnSwapNum',
        'Reduplication',
        'CnMLM',
        'SwapContraction',
        'SwapVerb',
        'SwapSyn'
    ],
    'NER': [
        'ConcatSent',
        'EntTypos',
        'SwapEnt'
    ],
    'POS': [
        'SwapMultiPOS',
        'SwapPrefix'
    ],
    'DP': [
        'DeleteSubTree',
        'AddSubTree'
    ],
    'MRC': [
        'AddSentDiverse',
        'ModifyPos',
        'PerturbAnswer',
        'PerturbQuestion'
    ],
    'SM': [
        'SwapWord',
        'SwapNum',
        'Overlap'
    ],
    'NLI': [
        'SwapAnt',
        'AddSent',
        'NumWord',
        'Overlap'
    ],
    'RE': [
        'InsertClause',
        'SwapAge',
        'SwapBirth',
        'SwapEmployee',
        'SwapEnt'
    ],
    'COREF': [
        'RndConcat',
        'RndDelete',
        'RndInsert',
        'RndRepeat',
        'RndReplace',
        'RndShuffle'
    ]
}

# indicate allowed transformations of specific task
ALLOWED_TRANSFORMATIONS = {
    key: list(set(TASK_TRANSFORMATIONS[key] + UT_TRANSFORMATIONS)
              ^ set(UNMATCH_UT_TRANSFORMATIONS[key]))
    for key in TASK_TRANSFORMATIONS
}

ALLOWED_VALIDATORS = {
    key: [
        "EditDistance",
        "GPT2Perplexity",
        "MaxWordsPerturbed",
        "SentenceEncoding",
        "TranslateScore"
    ]
    for key in TASK_TRANSFORMATIONS
}

# -------------------------NLTK settings---------------------------
DP_PARSER = "NLTK_DATA/parser/parser.zip"
CFG_PARSER = "NLTK_DATA/parser/parser.zip"
NLTK_WORDNET = "NLTK_DATA/wordnet/wordnet.zip"
NLTK_WORDNET_DELEMMA = "NLTK_DATA/wordnet/word_delemma.pkl"
STANFORD_NER = "NLTK_DATA/ner/ner.zip"
STANFORD_POS_TAGGER = "NLTK_DATA/tagger/tagger.zip"

# -------------------------UT settings---------------------------
STOP_WORDS = [
    'I', 'my', 'My', 'mine', 'you', 'your', 'You', 'Your',
    'He', 'he', 'him', 'Him', 'His', 'his', 'She', 'she',
    'her', 'Her', 'it', 'It', 'they', 'They', 'their',
    'Their', 'am', 'Are', 'are', 'Is', 'is', 'And', 'and',
    'or', 'nor', 'A', 'a', 'An', 'an', 'the', 'The', 'Have',
    'have', 'Has', 'has', 'in', 'In', 'by', 'By',
    'on', 'On', 'of', 'Of', 'at', 'At', 'from', 'From'
]

# sentence splitter prefixes
EN_NON_BREAKING_PRE = 'UT_DATA/en_non_breaking_prefixes.txt'

BERT_MODEL_NAME = 'bert-base-uncased'

# back translation model
TRANS_FROM_MODEL = "facebook/wmt19-en-de"  # "allenai/wmt16-en-de-dist-6-1"
TRANS_TO_MODEL = "facebook/wmt19-de-en"  # allenai/wmt19-de-en-6-6-base"

# Offline Vocabulary
EMBEDDING_PATH = 'UT_DATA/sim_words_dic.json'
VERB_PATH = 'UT_DATA/verb_tenses.json'

# SpellingError error vocabulary
SPELLING_ERROR_DIC = 'UT_DATA/spelling_en.txt'
# keyboard rules vocabulary
KEYBOARD_RULES = 'UT_DATA/keyboard/en.json'

ADVERB_PATH = 'UT_DATA/neu_adverb_word_228.txt'
TWITTER_PATH = 'UT_DATA/twitter_contraction.json'
BEGINNING_PATH = 'UT_DATA/beginning.txt'
PROVERB_PATH = 'UT_DATA/proverb.txt'
PREJUDICE_PATH = 'UT_DATA/names.json'
PREJUDICE_WORD_PATH = 'UT_DATA/prejudice.txt'
PREJUDICE_LOC_PATH = 'UT_DATA/loc2idx'
PREJUDICE_LOC2IDX = {
    'America': 1, 'Europe': 2, 'Africa': 3, 'China': 4,
    'Japan': 5, 'India': 6, 'Middle East': 7
}

ENTITIES_PATH = 'UT_DATA/lop_entities.json'

# Verb pos tag
VERB_TAG = ['VB', 'VBP', 'VBZ', 'VBG', 'VBD', 'VBN']

# UT word_contraction
CONTRACTION_PHRASES = {
    'is not': "isn't", 'are not': "aren't", 'cannot': "can't",
    'could not': "couldn't", 'did not': "didn't", 'does not': "doesn't",
    'do not': "don't", 'had not': "hadn't", 'has not': "hasn't",
    'have not': "haven't", 'might not': "mightn't", 'must not': "mustn't",
    'need not': "needn't", 'ought not': "oughtn't", 'shall not': "shan't",
    'should not': "shouldn't", 'was not': "wasn't", 'were not': "weren't",
    'will not': "won't", 'would not': "wouldn't",

    'I will': "I'll", 'i will': "i'll", 'It is': "It's", 'it is': "it's",
    'I am': "I'm", 'i am': "i'm", 'I would': "I'd", 'i would': "i'd",
    'it would': "it'd", 'It would': "It'd", 'it will': "it'll",
    'It will': "It'll", 'she will': "she'll", 'She will': "She'll",
    'she is': "she's", 'She is': "She's", 'they would': "they'd",
    'They would': "They'd", 'they will': "they'll", 'They will': "They'll",
    'they are': "they're", 'They are': "They're", 'we would': "we'd",
    'We would': "We'd", 'we will': "we'll", 'We will': "We'll",
    'we are': "we're", 'We are': "We're", 'she would': "she'd",
    'She would': "She'd", 'you would': "you'd", 'You would': "You'd",
    'you will': "you'll", 'You will': "You'll", 'you are': "you're",
    'You are': "You're", 'he is': "he's", 'He is': "He's",

    'that would': "that'd", 'That would': "That'd", 'that is': "that's",
    'That is': "That's", 'there would': "there'd", 'There would': "There'd",
    'there is': "there's", 'There is': "There's", 'how did': "how'd",
    'How did': "How'd", 'how is': "how's", 'How is': "How's",
    'what are': "what're", 'What are': "What're", 'what is': "what's",
    'What is': "What's", 'when is': "when's", 'When is': "When's",
    'where did': "where'd", 'Where did': "Where'd", 'where is': "where's",
    'Where is': "Where's", 'who will': "who'll", 'Who will': "Who'll",
    'who is': "who's", 'Who is': "Who's", 'who have': "who've",
    'Who have': "Who've", 'why is': "why's", 'Why is': "why's"
}

# UT sent_add_sent
MIN_SENT_TRANS_LENGTH = 10

# corenlp entity map to LPO
CORENLP_ENTITY_MAP = {
    'LOCATION': 'LOCATION',
    'PERSON': 'PERSON',
    'ORGANIZATION': 'ORGANIZATION',
    'COUNTRY': 'LOCATION',
    'STATE_OR_PROVINCE': 'LOCATION',
    'CITY': 'LOCATION',
    'NATIONALITY': 'LOCATION'
}
MODEL_PATH_WEB = 'SPACY_MODEL/model.zip'
MODEL_PATH = os.path.normcase('/en_core_web_lg/en_core_web_lg-3.0.0')

# ---------------------------CWS settings---------------------------
CWS_DATA_PATH = 'CWS_DATA/'
NUM_LIST = [
    '零', '一', '二', '三', '四', '五', '六',
    '七', '八', '九', '十', '百', '千', '万', '亿'
]
# Judge whether the number is less than ten
NUM_FLAG1 = 9
# Judge whether the number is less than ten thousand
NUM_FLAG2 = 12
# number begin
NUM_BEGIN = 1
# number end
NUM_END = 9
abbreviation_path = CWS_DATA_PATH + 'abbreviation'
NAME_PATH = CWS_DATA_PATH + '姓.txt'
WORD_LIST_PATH = CWS_DATA_PATH + 'dict'
ABAB_PATH = CWS_DATA_PATH + '60ABAB'
AABB_PATH = CWS_DATA_PATH + '650AABB'
AONEA_PATH = CWS_DATA_PATH + 'A-one-A'
SYNONYM_PATH = CWS_DATA_PATH + 'synonym'
DETACHABLE_WORD_PATH = CWS_DATA_PATH + 'detachable_word'

# ---------------------- NLI & SM settings -----------------------------------
BLACK_LIST_WORD = [
    "here", "goodness", "yes", "no", "decision", "growing",
    "priority", "cheers", "volume", "right", "left", "goods",
    "addition", "income", "indecision", "there", "parent",
    "being", "parents", "lord", "lady", "put", "capital",
    "lowercase", "unions"
]

# -------------------------SA settings---------------------------
SA_PERSON_PATH = 'SA_DATA/person_info.csv'
SA_MOVIE_PATH = 'SA_DATA/movie_info.csv'

SA_DOUBLE_DENIAL_DICT = {
    'poor': 'not good', 'bad': 'not great', 'lame': 'not interesting',
    'awful': 'not awesome', 'great': 'not bad', 'good': 'not poor',
    'applause': 'not discourage',
    'recommend': "don't prevent", 'best': 'not worst',
    'encourage': "don't discourage",
    'entertain': "don't disapprove",
    'wonderfully': 'not poorly', 'love': "don't hate",
    'interesting': "not uninteresting", 'interested': 'not ignorant',
    'glad': 'not reluctant', 'positive': 'not negative',
    'perfect': 'not imperfect',
    'entertaining': 'not uninteresting',
    'moved': 'not moved', 'like': "don't refuse",
    'worth': 'not undeserving',
    'better': 'not worse', 'funny': 'not uninteresting',
    'awesome': 'not ugly',
    'impressed': 'not impressed'
}

# -------------------------POS settings---------------------------
MORPHEME_ANALYZER = 'POS_DATA/en.morph.tar.bz2'

# -------------------------ABSA settings---------------------------
NEGATIVE_WORDS_LIST = [
    'doesn\'t', 'don\'t', 'didn\'t', 'no', 'did not', 'do not',
    'does not', 'not yet', 'not', 'none', 'no one', 'nobody', 'nothing',
    'neither', 'nowhere', 'never', 'hardly', 'scarcely', 'barely'
]
DEGREE_WORD_LIST = [
    'absolutely', 'awfully', 'badly', 'barely', 'completely', 'decidedly',
    'deeply', 'enormously', 'entirely', 'extremely', 'fairly', 'fully',
    'greatly', 'highly', 'incredibly', 'indeed', 'very', 'really'
]
PHRASE_LIST = [
    'ASJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC',
    'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP',
    'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X', 'S', 'SBAR'
]
ABSA_DATA_PATH = 'ABSA_DATA/'
ABSA_TRAIN_RESTAURANT_PATH = 'ABSA_DATA/restaurant/train_sent_towe.json'
ABSA_TRAIN_LAPTOP_PATH = 'ABSA_DATA/laptop/train_sent_towe.json'

# -------------------------NER settings---------------------------
NER_OOV_ENTITIES = 'NER_DATA/OOVentities'
LONG_ENTITIES = 'NER_DATA/long_dict.json'
CROSS_ENTITIES = 'NER_DATA/Multientity'

# ---------------------------DP settings---------------------------
WIKIDATA_STATEMENTS = [
    'P101', 'P19', 'P69', 'P800', 'P1066', 'P50',
    'P57', 'P136', 'P921', 'P159', 'P740', 0
]

CLAUSE_HEAD = {
    'P101': 'who worked at ',
    'P19': 'who was born in ',
    'P69': 'who was educated at ',
    'P800': 'whose notable work is ',
    'P1066': 'who is the student of ',
    'P136': 'which is a ',
    'P57': 'which is directed by ',
    'P921': 'with the subject of the ',
    'P50': 'which is written by ',
    'P159': 'headquartered in ',
    'P740': 'established in ',
    0: 'which is a '
}

WIKIDATA_INSTANCE = {
    'instance': 'P31',
    'disambiguation': 'Q4167410',
    'human': 'Q5'
}

# ---------------------------RE settings---------------------------
WIKIDATA_STATEMENTS_no_zero = [
    'P101', 'P19', 'P69', 'P800', 'P1066',
    'P50', 'P57', 'P136', 'P921', 'P159', 'P740'
]
scbase = 'https://www.wikidata.org/w/api.php?act' \
         'ion=wbsearchentities&language=en&format=json&limit=3&search='
CLAUSE_HEAD_no_zero = {
    'P101': 'who worked at ',
    'P19': 'who was born in ',
    'P69': 'who was educated at ',
    'P800': 'whose notable work is ',
    'P1066': 'who is the student of ',
    'P136': 'which is a ',
    'P57': 'which is directed by ',
    'P921': 'with the subject of the ',
    'P50': 'which is written by ',
    'P159': 'headquartered in ',
    'P740': 'established in ',
    'description': 'which is a '
}

LOWFREQ = "RE_DATA/lowfreq.json"
TYPES = "RE_DATA/types.json"
MULTI_TYPE = "RE_DATA/multi_type.json"
TITLE = "RE_DATA/titles.json"

# ---------------------------MRC settings---------------------------
NEAR_DICT_PATH = 'MRC_DATA/neighbour.json'
POS_DICT_PATH = 'MRC_DATA/postag_dict.json'

# ---------------------------Subpopulation settings---------------------------
NEGATION = [
    "no", "not", "none", "noone ", "nobody", "nothing",
    "neither", "nowhere", "never", "hardly", "scarcely",
    "barely", "doesnt", "isnt", "wasnt", "shouldnt",
    "wouldnt", "couldnt", "wont", "cant", "dont"
]
QUESTION = [
    "what", "how", "why", "when", "where", "who", "whom"
]
