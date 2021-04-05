
DEFAULT_CONFIG = {
  "task": None,
  "max_trans": 1,
  "semantic_validate": False,
  "semantic_score": 0.7,
  "fields": "x",
  "keep_origin": False,
  "return_unk": True,
  "task_config": {
  },
  "transform_methods": [],
  "subpopulation_methods": [
    "LengthSubPopulation",
    "PhraseSubPopulation",
    "LMSubPopulation",
    "PrejudiceSubPopulation"
  ]
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

TASK_CONFIGS = {
    "WordCase":  [
      {"case_type": "upper"},
      {"case_type": "lower"},
      {"case_type": "title"}
    ],
    'MLM': {
        "device": "cuda:0"
    },
    'BackTrans': {
        "device": "cuda:0"
    },
    'Prejudice': [{"change_type": "Name", "prejudice_tendency": "man"},
                  {"change_type": "Name", "prejudice_tendency": "woman"},
                  {"change_type": "Loc", "prejudice_tendency": ['America', 'Europe', 'China', 'Japan']},
                  {"change_type": "Loc", "prejudice_tendency": ['Africa', 'India', 'Middle East']}],
    "SwapMultiPOS": [
      {"treebank_tag": "NN"},
      {"treebank_tag": "RB"},
      {"treebank_tag": "JJ"},
      {"treebank_tag": "VB"}
    ],
    "AddWordNet": [
      {"treebank_tags": ["VB", "VBP", "VBZ", "VBG", "VBD", "VBN"], "add_treebank_tag":  "RB"},
      {"treebank_tags": ["NN", "NNP", "NNS", "NNPS"], "add_treebank_tag":  "JJ"}
    ],
    "AddSummaryCSV": [
      {"name_type": "movie"},
      {"name_type": "person"}
    ],
    "SpecialEntityReplace": [
      {"name_type": "movie"}, 
      {"name_type": "person"}
    ],
    'PerturbAnswer': [
        {'transform_method': 'BackTrans'},
        {'transform_method': 'MLM'}
    ],
    "LengthSubPopulation": [
      {"intervals": ["80%", "100%"]},
      {"intervals": ["0%", "20%"]}
    ],
    "LMSubPopulation": [
      {"intervals": ["80%", "100%"], "device": "cuda:1"},
      {"intervals": ["0%", "20%"], "device": "cuda:1"}
    ],
    "PhraseSubPopulation": [
      {"phrase_name": "negation"},
      {"phrase_name": "question"}
    ],
    "PrejudiceSubPopulation": [
      {"mode": "man"},
      {"mode": "woman"}
    ]
}

TRANSFORMATIONS = {
    'UT': [
        'AddAdverb',
        'AppendIrr',
        'BackTrans',
        'WordCase',
        'Contraction',
        'Entity',
        'Keyboard',
        'MLM',
        'Number',
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
    ],
    'ABSA': [
        'AbsaReverseTarget',
        'AbsaReverseNonTarget',
        'AbsaAddDiff',
    ],
    'SA': [
        'SpecialWordDoubleDenial',
        'SpecialEntityReplace',
        'AddSummaryCSV'
    ],
    'NER': [
        'ConcatCase',
        'CrossCategory',
        'EntityTyposSwap',
        'OOV',
        'ToLonger'
    ],
    'POS': [
        'SwapMultiPOS',
        'SwapPrefix'
    ],
    'DP': [
        'RemoveSubtree',
        'AddSubtree'
    ],
    'MRC': [
      'ModifyPosition',
      'PerturbAnswer'
      'AddSentenceDiverse',
      'PerturbQuestion'
    ],
    'SM': [
        'SmAntonymSwap',
        'SmNumWord',
        'SmOverlap'
    ],
    'NLI': [
        'NliAntonymSwap',
        'NliLength',
        'NliNumWord',
        'NliOverlap'
    ],
    'RE': [
        'AddClause',
        'AgeSwap',
        'BirthSwap',
        'EmployeeSwap',
        'LowFreqSwap',
        'MultiType',
        'SameEtypeSwap'
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

# 先开始的变形不影响后续变形，在满足这个条件的前提下，变形条件苛刻的置于前
TRANSFORMATION_PRIORITY = {
    # Coref transformations

    'RandomDelete': 3.70,
    'RandomShuffle': 3.71,
    'RandomReplace': 3.72,
    'RandomRepeat': 3.73,
    'RandomInsert': 3.74,
    'RandomConcat': 3.75,

    # UT transformations
    'WordCase': 5.0,

    'Prejudice': 4.4,
    'Typos': 4.3,
    'Keyboard': 4.2,
    'Ocr': 4.1,
    'SpellingError': 4.0,

    'Punctuation': 3.6,
    'AppendIrr': 3.5,
    'TwitterType': 3.4,
    'SwapSynWordEmbedding': 3.3,
    'Tense': 3.3,
    'AddAdverb': 3.2,
    'SwapSynWordNet': 3.1,
    'Contraction': 3.0,

    'MLM': 2.2,
    'ReverseNeg': 2.1,
    'BackTrans': 2.0,

    'SwapAntWordNet': 1.2,
    'Entity': 1.1,
    'Number': 1.0,

    "LengthSubPopulation":0.1,
    "PhraseSubPopulation":0.1,
    "LMSubPopulation":0.1,
    "PrejudiceSubPopulation":0.1

}

NOT_ALLOWED_UT_TRANS = {
    'ABSA': ['ReverseNeg', 'BackTrans', 'SwapAntWordNet'],
    'SA': ['ReverseNeg', 'SwapAntWordNet'],
    'NER': ['BackTrans', 'Entity', 'Prejudice'],
    'POS': ['BackTrans', 'Tense'],
    'DP': ['BackTrans'],
    'MRC': ['Number', 'ReverseNeg', 'Entity', 'SwapAntWordNet', 'BackTrans', 'MLM', 'Prejudice'],
    'SM': ['SwapAntWordNet', 'ReverseNeg'],
    'NLI': ['SwapAntWordNet', 'ReverseNeg'],
    'RE': ['BackTrans', 'Entity', 'Number', 'ReverseNeg', 'Prejudice'],
    'COREF': ['BackTrans']
}
