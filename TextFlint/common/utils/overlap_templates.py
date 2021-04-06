import random
import numpy as np

""" Overlap templates for NLI transformation NLIOverlap
    Heavily borrowed from Right for the Wrong Reasons: Diagnosing Syntactic 
    Heuristics in Natural Language Inference ACL2019.
    For code, please check the following link:
    https://github.com/tommccoy1/hans
"""

def despace(string):
    new_string = string.replace("  ", " ")

    if new_string == string:
        return string
    else:
        return despace(new_string)


def remove_terminals(tree):
    words = tree.split()
    new_words = []

    for word in words:
        if word[0] == "(":
            new_words.append("(")
        else:
            new_words.append(word)

    new_tree = " ".join(new_words)
    new_tree = new_tree.replace("(", " ( ").replace(")", " ) ")

    return despace(new_tree.strip())


def binarize_helper(tree):
    prev_word = ""

    for index, word in enumerate(tree):
        if word != "(" and word != ")":
            if prev_word == "(":
                if index < len(tree) - 1:
                    if tree[index + 1] == ")":
                        return binarize_helper(
                            tree[:index - 1] + [tree[index]] + tree[index + 2:])
        prev_word = word

    for index, word in enumerate(tree):
        if word != "(" and word != ")":
            if prev_word == "(":
                if index < len(tree) - 2:
                    if tree[index + 1] != ")" and tree[index + 1] \
                            != "(" and tree[index + 2] == ")":
                        return binarize_helper(
                            tree[:index - 1] + [" ".join(
                                tree[index - 1:index + 3])] + tree[index + 3:])
        prev_word = word

    for index, word in enumerate(tree):
        if word == ")":
            if index > 2:
                if tree[index - 1] != "(" and tree[index - 1] != ")" \
                    and tree[index - 2] != "(" and tree[index - 2] != ")" \
                        and tree[index - 3] != "(" and tree[index - 3] != ")":
                    return binarize_helper(
                        tree[:index - 2] +
                        ["( " + tree[index - 2] + " " + tree[index - 1] + " )"]
                        + tree[index:])

    return tree


def valid_tree(binary_tree):
    if len(binary_tree) > 1:
        return False

    tree = binary_tree[0]
    parts = tree.split()

    count_brackets = 0
    for part in parts:
        if part == "(":
            count_brackets += 1
        elif part == ")":
            count_brackets -= 1

        if count_brackets < 0:
            return False

    if count_brackets == 0:
        return True


def binarize_tree(tree):
    unterminaled = remove_terminals(tree)

    words = unterminaled.split()

    tree = binarize_helper(words)
    if not valid_tree(tree):
        print("WRONG TREE")

    return binarize_helper(words)[0]


def generate_vp():
    verb = random.choice(verbs)
    if verb in intransitive_verbs:
        return verb, "(VP (VBD " + verb + "))"
    else:
        obj = random.choice(nouns)

        return verb + " the " + \
            obj, "(VP (VBD " + verb + ") (NP (DT the) (NN " + obj + ")))"


def parse_vp(vp):
    words = vp.split()

    if len(words) == 1:
        return "(VP (VBD " + words[0] + "))"
    else:
        return "(VP (VBD " + words[0] + \
            ") (NP (DT the) (NN " + words[2] + ")))"


def parse_pp(pp):
    words = pp.split()

    if words[:2] == ["next", "to"]:
        return "(ADVP (JJ next) (PP (TO to) (NP (DT the) (" + \
            noun_tag(words[-1]) + " " + words[-1] + "))))"
    elif words[:3] == ["in", "front", "of"]:
        return "(PP (IN in) (NP (NP (NN front)) (PP (IN of) (NP (NP (DT the) (" \
               + noun_tag(words[-1]) + " " + words[-1] + "))))))"
    else:
        return "(PP (IN " + words[0] + ") (NP (DT the) (" + \
            noun_tag(words[-1]) + " " + words[-1] + ")))"


def generate_rc():
    rel = random.choice(rels)

    verb = random.choice(verbs)
    if verb in intransitive_verbs:
        return rel + " " + verb
    else:
        arg = random.choice(nouns)
        if random.randint(0, 1) == 0:
            return rel + " the " + arg + " " + verb
        else:
            return rel + " " + verb + " the " + arg


def noun_tag(noun):
    if noun in nouns_sg or noun in told_objects or noun in food_words \
            or noun in location_nouns or noun in location_nouns_b \
            or noun in won_objects or noun in read_wrote_objects:
        return "NN"
    elif noun in nouns_pl:
        return "NNS"
    else:
        print("INVALID WORD", noun)


def parse_rc(rc):
    words = rc.split()

    if words[0] == "that":
        if len(words) == 2:
            return "(SBAR (WHNP (WDT that)) (S (VP (VBD " + words[1] + "))))"
        else:
            if words[1] == "the":
                return "(SBAR (WHNP (WDT that)) (S (NP (DT the) (" + \
                    noun_tag(words[2]) + " " + words[2] + ")) (VP (VBD " \
                       + words[3] + "))))"
            else:
                return "(SBAR (WHNP (WDT that)) (S (VP (VBD " + \
                    words[1] + ") (NP (DT the) (" + noun_tag(words[3]) \
                       + " " + words[3] + ")))))"

    elif words[0] == "who":
        if len(words) == 2:
            return "(SBAR (WHNP (WP who)) (S (VP (VBD " + words[1] + "))))"
        else:
            if words[1] == "the":
                return "(SBAR (WHNP (WP who)) (S (NP (DT the) (" + \
                    noun_tag(words[2]) + " " + words[2] + ")) (VP (VBD " \
                       + words[3] + "))))"
            else:
                return "(SBAR (WHNP (WP who)) (S (VP (VBD " + \
                    words[1] + ") (NP (DT the) (" + noun_tag(words[3]) + " " \
                       + words[3] + ")))))"

    else:
        print("INVALID RELATIVIZER")


def postprocess(sentence):
    sentence = sentence[0].upper() + sentence[1:]

    return sentence


def template_filler(template_list):
    probs = []
    templates = []

    for template_pair in template_list:
        probs.append(template_pair[0])
        templates.append(template_pair[1])

    template_index = np.random.choice(range(len(templates)), p=probs)
    template_tuple = templates[template_index]
    template = template_tuple[0]
    hypothesis_template = template_tuple[1]
    template_tag = template_tuple[2]

    premise_list = []
    index_dict = {}

    for (index, element) in template:
        if element == "VP":
            vp, vp_parse = generate_vp()
            premise_list.append(vp)
            index_dict[index] = vp

        elif element == "RC":
            rc = generate_rc()
            premise_list.append(rc)
            index_dict[index] = rc

        elif "vobj" in element:
            obj = random.choice(object_dict[index_dict[int(element[-1])]])
            premise_list.append(obj)
            index_dict[index] = obj

        elif isinstance(element, str):
            premise_list.append(element)
            index_dict[index] = element

        else:
            word = random.choice(element)
            premise_list.append(word)
            index_dict[index] = word

    hypothesis_list = [index_dict[ind] for ind in hypothesis_template]

    return postprocess(" ".join(premise_list)), postprocess(
        " ".join(hypothesis_list)), template_tag


nouns_sg = [
    "professor",
    "student",
    "president",
    "judge",
    "senator",
    "secretary",
    "doctor",
    "lawyer",
    "scientist",
    "banker",
    "tourist",
    "manager",
    "artist",
    "author",
    "actor",
    "athlete"]
nouns_pl = [
    "professors",
    "students",
    "presidents",
    "judges",
    "senators",
    "secretaries",
    "doctors",
    "lawyers",
    "scientists",
    "bankers",
    "tourists",
    "managers",
    "artists",
    "authors",
    "actors",
    "athletes"]
nouns = nouns_sg + nouns_pl
transitive_verbs = [
    "recommended",
    "called",
    "helped",
    "supported",
    "contacted",
    "believed",
    "avoided",
    "advised",
    "saw",
    "stopped",
    "introduced",
    "mentioned",
    "encouraged",
    "thanked",
    "recognized",
    "admired"]
passive_verbs = [
    "recommended",
    "helped",
    "supported",
    "contacted",
    "believed",
    "avoided",
    "advised",
    "stopped",
    "introduced",
    "mentioned",
    "encouraged",
    "thanked",
    "recognized",
    "admired"]
intransitive_verbs = [
    "slept",
    "danced",
    "ran",
    "shouted",
    "resigned",
    "waited",
    "arrived",
    "performed"]
verbs = transitive_verbs + intransitive_verbs

nps_verbs = ["believed", "knew", "heard"]
# "forgot", "preferred", "claimed", "wanted", "needed", "found", "suggested",
# "expected"] # These all appear at least 100 times with both NP and S arguments
npz_verbs = ["hid", "moved", "presented", "paid", "studied", "stopped"]
# "paid",  "changed", "studied", "answered", "stopped", "grew", "moved",
# "returned", "left","improved", "lost", "visited", "ate", "played"]
# All appear at least 100 times with both transitive and intransitive frames
npz_verbs_plural = ["fought"]
# All appear at least 100 times with both transitive and intransitive frames
understood_argument_verbs = [
    "paid",
    "explored",
    "won",
    "wrote",
    "left",
    "read",
    "ate"]
nonentailing_quot_vebs = [
    "hoped",
    "claimed",
    "thought",
    "believed",
    "said",
    "assumed"]
question_embedding_verbs = [
    "wondered",
    "understood",
    "knew",
    "asked",
    "explained",
    "realized"]

# Each appears at least 100 times in MNLI
preps = ["near", "behind", "by", "in front of", "next to"]
conjs = ["while", "after", "before", "when", "although", "because", "since"]
past_participles = ["studied", "paid", "helped", "investigated", "presented"]
called_objects = ["coward", "liar", "hero", "fool"]
told_objects = ["story", "lie", "truth", "secret"]
food_words = [
    "fruit",
    "salad",
    "broccoli",
    "sandwich",
    "rice",
    "corn",
    "ice cream"]
location_nouns = [
    "neighborhood",
    "region",
    "country",
    "town",
    "valley",
    "forest",
    "garden",
    "museum",
    "desert",
    "island",
    "town"]
location_nouns_b = ["museum", "school", "library", "office", "laboratory"]
won_objects = [
    "race",
    "contest",
    "war",
    "prize",
    "competition",
    "election",
    "battle",
    "award",
    "tournament"]
read_wrote_objects = [
    "book",
    "column",
    "report",
    "poem",
    "letter",
    "novel",
    "story",
    "play",
    "speech"]
adjs = ["important", "popular", "famous", "young", "happy",
        "helpful", "serious", "angry"]  # All at least 100 times in MNLI
adj_comp_nonent = ["afraid", "sure", "certain"]
adj_comp_ent = ["sorry", "aware", "glad"]
advs = ["quickly", "slowly", "happily", "easily", "quietly",
        "thoughtfully"]  # All at least 100 times in MNLI
const_adv = [
    "after",
    "before",
    "because",
    "although",
    "though",
    "since",
    "while"]
const_quot_entailed = ["forgot", "learned", "remembered", "knew"]
advs_nonentailed = ["supposedly", "probably", "maybe", "hopefully"]
advs_entailed = ["certainly", "definitely", "clearly", "obviously", "suddenly"]
rels = ["who", "that"]
quest = ["why", "how"]
nonent_complement_nouns = ["feeling", "evidence", "idea", "belief"]
ent_complement_nouns = ["fact", "reason", "news", "time"]
object_dict = {}
object_dict["called"] = called_objects
object_dict["told"] = told_objects
object_dict["brought"] = food_words
object_dict["made"] = food_words
object_dict["saved"] = food_words
object_dict["offered"] = food_words
object_dict["paid"] = nouns
object_dict["explored"] = location_nouns
object_dict["won"] = won_objects
object_dict["wrote"] = read_wrote_objects
object_dict["left"] = location_nouns
object_dict["read"] = read_wrote_objects
object_dict["ate"] = food_words

advs_embed_not_entailed = ["if", "unless"]
advs_embed_entailed = [
    "after",
    "before",
    "because",
    "although",
    "though",
    "since",
    "while"]
advs_outside_not_entailed = ["if", "unless"]
advs_outside_entailed = [
    "after",
    "before",
    "because",
    "although",
    "though",
    "since",
    "while"]

# Lexical Overlap: Simple sentence
lex_simple_templates = [(1.0, (
    [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
     (4, nouns), (5, ".")], [3, 4, 2, 0, 1, 5], "temp1",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1,
     ")) (VP (VBD ", 2, ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4,
     ")) (VP (VBD ", 2, ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"]))]

# Lexical Overlap: Preposition on subject
lex_prep_templates = [
    (1.0 / 6, (
        [(0, "the"), (1, nouns), (2, preps), (3, "the"),
         (4, nouns), (5, transitive_verbs), (6, "the"), (7, nouns),
         (8, ".")], [3, 4, 5, 0, 1, 8], "temp2",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ",
         "ppp,2,4", ") (VP (VBD ", 5, ") (NP (DT the) (", "nn,7", " ",
         7, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 5,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 6, (
        [(0, "the"), (1, nouns), (2, preps), (3, "the"), (4, nouns),
         (5, transitive_verbs), (6, "the"), (7, nouns),
         (8, ".")], [6, 7, 5, 0, 1, 8], "temp3",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "ppp,2,4",
         ") (VP (VBD ", 5, ") (NP (DT the) (", "nn,7", " ",
         7, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,7", " ", 7, ")) (VP (VBD ", 5,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 6, (
        [(0, "the"), (1, nouns), (2, preps), (3, "the"), (4, nouns),
         (5, transitive_verbs), (6, "the"), (7, nouns),
         (8, ".")], [6, 7, 5, 3, 4, 8], "temp4",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "ppp,2,4",
         ") (VP (VBD ", 5, ") (NP (DT the) (", "nn,7", " ",
         7, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,7", " ", 7, ")) (VP (VBD ", 5,
         ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"])),
    (1.0 / 6, (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
         (4, nouns), (5, preps), (6, "the"), (7, nouns),
         (8, ".")], [3, 4, 2, 0, 1, 8], "temp5",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "ppp,5,7", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 6, (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
         (4, nouns), (5, preps), (6, "the"), (7, nouns),
         (8, ".")], [3, 4, 2, 6, 7, 8], "temp6",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "ppp,5,7", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,7", " ", 7, "))) (. .)))"])),
    (1.0 / 6, (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
         (4, nouns), (5, preps), (6, "the"), (7, nouns),
         (8, ".")], [6, 7, 2, 0, 1, 8], "temp7",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "ppp,5,7", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,7", " ", 7, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"]))
]

# Lexical Overlap: Relative clause on subject
lex_rc_templates = [
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, rels), (3, "the"), (4, nouns),
         (5, transitive_verbs), (6, transitive_verbs),
         (7, "the"), (8, nouns), (9, ".")], [3, 4, 6, 0, 1, 9], "temp8",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,5",
         ") (VP (VBD ", 6, ") (NP (DT the) (", "nn,8", " ",
         8, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 6,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, rels), (3, "the"), (4, nouns),
         (5, transitive_verbs), (6, transitive_verbs),
         (7, "the"), (8, nouns), (9, ".")], [7, 8, 6, 0, 1, 9], "temp9",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,5",
         ") (VP (VBD ", 6, ") (NP (DT the) (", "nn,8", " ",
         8, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,8", " ", 8, ")) (VP (VBD ", 6,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, rels), (3, "the"), (4, nouns),
         (5, transitive_verbs), (6, transitive_verbs),
         (7, "the"), (8, nouns), (9, ".")], [7, 8, 6, 3, 4, 9], "temp10",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,5",
         ") (VP (VBD ", 6, ") (NP (DT the) (", "nn,8", " ",
         8, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,8", " ", 8, ")) (VP (VBD ", 6,
         ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, rels), (3, transitive_verbs),
         (4, "the"), (5, nouns), (6, transitive_verbs),
         (7, "the"), (8, nouns), (9, ".")], [4, 5, 6, 0, 1, 9], "temp11",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,5",
         ") (VP (VBD ", 6, ") (NP (DT the) (", "nn,8", " ",
         8, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,5", " ", 5, ")) (VP (VBD ", 6,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, rels), (3, transitive_verbs),
         (4, "the"), (5, nouns), (6, transitive_verbs),
         (7, "the"), (8, nouns), (9, ".")], [7, 8, 6, 0, 1, 9], "temp12",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,5",
         ") (VP (VBD ", 6, ") (NP (DT the) (", "nn,8", " ",
         8, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,8", " ", 8, ")) (VP (VBD ", 6,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, rels), (3, transitive_verbs),
         (4, "the"), (5, nouns), (6, transitive_verbs),
         (7, "the"), (8, nouns), (9, ".")], [7, 8, 6, 4, 5, 9], "temp13",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,5",
         ") (VP (VBD ", 6, ") (NP (DT the) (", "nn,8", " ",
         8, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,8", " ", 8, ")) (VP (VBD ", 6,
         ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"), (4, nouns),
         (5, rels), (6, transitive_verbs),
         (7, "the"), (8, nouns), (9, ".")], [3, 4, 2, 0, 1, 9], "temp14",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "prc,5,8", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
         (4, nouns), (5, rels), (6, transitive_verbs),
         (7, "the"), (8, nouns), (9, ".")], [3, 4, 2, 0, 7, 8, 9], "temp15",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "prc,5,8", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,8", " ", 8, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"), (4, nouns),
         (5, rels), (6, transitive_verbs),
         (7, "the"), (8, nouns), (9, ".")], [7, 8, 2, 0, 1, 9], "temp16",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "prc,5,8", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,8", " ", 8, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"), (4, nouns),
         (5, rels), (6, "the"), (7, nouns),
         (8, transitive_verbs), (9, ".")], [3, 4, 2, 0, 1, 9], "temp17",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "prc,5,8", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"), (4, nouns),
         (5, rels), (6, "the"), (7, nouns),
         (8, transitive_verbs), (9, ".")], [3, 4, 2, 6, 7, 9], "temp18",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "prc,5,8", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,7", " ", 7, "))) (. .)))"])),
    (1.0 / 12, (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"), (4, nouns),
         (5, rels), (6, "the"), (7, nouns),
         (8, transitive_verbs), (9, ".")], [6, 7, 2, 0, 1, 9], "temp19",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "prc,5,8", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,7", " ", 7, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"]))
]

# Lexical Overlap: Passive incorrect
lex_pass_templates = [
    (0.5, ([(0, "the"), (1, nouns_sg), (2, "was"), (3, passive_verbs),
            (4, "by"), (5, "the"), (6, nouns), (7, ".")],
           [0, 1, 3, 5, 6, 7], "temp20",
           ["(ROOT (S (NP (DT The) (NN ", 1, ")) (VP (VBD was) (VP (VBN ", 3,
            ") (PP (IN by) (NP (DT the) (", "nn,6",
            " ", 6, "))))) (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 3,
            ") (NP (DT the) (", "nn,6", " ", 6,
            "))) (. .)))"])),
    (0.5, ([(0, "the"), (1, nouns_pl), (2, "were"), (3, passive_verbs),
            (4, "by"), (5, "the"), (6, nouns), (7, ".")],
           [0, 1, 3, 5, 6, 7], "temp21",
           ["(ROOT (S (NP (DT The) (NNS ", 1, ")) (VP (VBD were) (VP (VBN ", 3,
            ") (PP (IN by) (NP (DT the) (", "nn,6",
            " ", 6, "))))) (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 3,
            ") (NP (DT the) (", "nn,6", " ", 6,
            "))) (. .)))"]))
]

# Lexical Overlap: Conjunctions
lex_conj_templates = [
    (0.25, ([(0, "the"), (1, nouns), (2, "and"), (3, "the"), (4, nouns),
             (5, transitive_verbs), (6, "the"), (7, nouns),
             (8, ".")], [0, 1, 5, 3, 4, 8], "temp22",
            ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1,
             ")) (CC and) (NP (DT the) (", "nn,4", " ", 4,
             "))) (VP (VBD ", 5, ") (NP (DT the) (", "nn,7",
             " ", 7, "))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 5,
             ") (NP (DT the) (", "nn,4", " ", 4,
             "))) (. .)))"])),
    (0.25, ([(0, "the"), (1, nouns), (2, "and"), (3, "the"), (4, nouns),
             (5, transitive_verbs), (6, "the"), (7, nouns),
             (8, ".")], [3, 4, 5, 0, 1, 8], "temp23",
            ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1,
             ")) (CC and) (NP (DT the) (", "nn,4", " ", 4,
             "))) (VP (VBD ", 5, ") (NP (DT the) (", "nn,7", " ", 7,
             "))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 5,
             ") (NP (DT the) (", "nn,1", " ", 1,
             "))) (. .)))"])),
    (0.25, ([(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
             (4, nouns), (5, "and"), (6, "the"), (7, nouns),
             (8, ".")], [3, 4, 2, 6, 7, 8], "temp24",
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
             ") (NP (NP (DT the) (", "nn,4", " ", 4,
             ")) (CC and) (NP (DT the) (", "nn,7", " ", 7, ")))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
             ") (NP (DT the) (", "nn,7", " ", 7,
             "))) (. .)))"])),
    (0.25, ([(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
             (4, nouns), (5, "and"), (6, "the"), (7, nouns),
             (8, ".")], [6, 7, 2, 3, 4, 8], "temp25",
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
             ") (NP (NP (DT the) (", "nn,4", " ", 4,
             ")) (CC and) (NP (DT the) (", "nn,7", " ", 7, ")))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,7", " ", 7, ")) (VP (VBD ", 2,
             ") (NP (DT the) (", "nn,4", " ", 4,
             "))) (. .)))"]))
]

# Lexical Overlap: Relative clause
lex_rc_ent_templates = [
    (0.25, ([(0, "the"), (1, nouns), (2, rels), (3, "the"), (4, nouns),
             (5, transitive_verbs), (6, transitive_verbs),
             (7, "the"), (8, nouns), (9, ".")], [3, 4, 5, 0, 1, 9], "temp26",
            ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,5",
             ") (VP (VBD ", 6, ") (NP (DT the) (",
             "nn,8", " ", 8, "))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 5,
             ") (NP (DT the) (", "nn,1", " ", 1,
             "))) (. .)))"])),
    (0.25, ([(0, "the"), (1, nouns), (2, rels), (3, transitive_verbs),
             (4, "the"), (5, nouns), (6, transitive_verbs),
             (7, "the"), (8, nouns), (9, ".")], [0, 1, 3, 4, 5, 9], "temp27",
            ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,5",
             ") (VP (VBD ", 6, ") (NP (DT the) (",
             "nn,8", " ", 8, "))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 3,
             ") (NP (DT the) (", "nn,5", " ", 5,
             "))) (. .)))"])),
    (0.25, ([(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
             (4, nouns), (5, rels), (6, transitive_verbs),
             (7, "the"), (8, nouns), (9, ".")], [3, 4, 6, 7, 8, 9], "temp28",
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
             ") (NP (NP (DT the) (", "nn,4", " ", 4,
             ")) ", "prc,5,8", ")) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 6,
             ") (NP (DT the) (", "nn,8", " ", 8,
             "))) (. .)))"])),
    (0.25, ([(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
             (4, nouns), (5, rels), (6, "the"), (7, nouns),
             (8, transitive_verbs), (9, ".")], [6, 7, 8, 3, 4, 9], "temp29",
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
             ") (NP (NP (DT the) (", "nn,4", " ", 4,
             ")) ", "prc,5,8", ")) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,7", " ", 7, ")) (VP (VBD ", 8,
             ") (NP (DT the) (", "nn,4", " ", 4,
             "))) (. .)))"]))
]

# Lexical Overlap: Across PP
lex_cross_pp_ent_templates = [(1.0, (
    [(0, "the"), (1, nouns), (2, preps), (3, "the"), (4, nouns),
     (5, transitive_verbs), (6, "the"), (7, nouns), (8, ".")],
    [0, 1, 5, 6, 7, 8], "temp30",
    ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "ppp,2,4",
     ") (VP (VBD ", 5, ") (NP (DT the) (", "nn,7", " ", 7,
     "))) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 5,
     ") (NP (DT the) (", "nn,7", " ", 7, "))) (. .)))"]))]

# Lexical Overlap: Across relative clause
lex_cross_rc_ent_templates = [(1.0, (
    [(0, "the"), (1, nouns), (2, "RC"), (3, transitive_verbs), (4, "the"),
     (5, nouns), (6, ".")], [0, 1, 3, 4, 5, 6],
    "temp31",
    ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,2",
     ") (VP (VBD ", 3, ") (NP (DT the) (", "nn,5", " ", 5,
     "))) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 3,
     ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"]))]

# Lexical Overlap: Conjunctions
lex_ent_conj_templates = [
    (0.5, ([(0, "the"), (1, nouns), (2, "and"), (3, "the"), (4, nouns),
            (5, transitive_verbs), (6, "the"), (7, nouns),
            (8, ".")], [0, 1, 5, 6, 7, 8], "temp32",
           ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1,
            ")) (CC and) (NP (DT the) (", "nn,4", " ", 4,
            "))) (VP (VBD ", 5, ") (NP (DT the) (", "nn,7", " ", 7,
            "))) (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 5,
            ") (NP (DT the) (", "nn,7", " ", 7,
            "))) (. .)))"])),
    (0.5, ([(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
            (4, nouns), (5, "and"), (6, "the"), (7, nouns),
            (8, ".")], [0, 1, 2, 6, 7, 8], "temp33",
           ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
            ") (NP (NP (DT the) (", "nn,4", " ", 4,
            ")) (CC and) (NP (DT the) (", "nn,7", " ", 7, ")))) (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
            ") (NP (DT the) (", "nn,7", " ", 7,
            "))) (. .)))"]))
]

# Lexical Overlap: Across adverb
lex_cross_adv_ent_templates = [
    (1.0, ([
        (0, "the"), (1, nouns), (2, advs), (3, verbs), (4, ".")], [
            0, 1, 3, 4], "temp34"))]

# Lexical Overlap: Passive
lex_ent_pass_templates = [
    (0.5, ([(0, "the"), (1, nouns_sg), (2, "was"), (3, passive_verbs),
            (4, "by"), (5, "the"), (6, nouns), (7, ".")],
           [5, 6, 3, 0, 1, 7], "temp35",
           ["(ROOT (S (NP (DT The) (NN ", 1, ")) (VP (VBD was) (VP (VBN ", 3,
            ") (PP (IN by) (NP (DT the) (", "nn,6",
            " ", 6, "))))) (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) (VP (VBD ", 3,
            ") (NP (DT the) (", "nn,1", " ", 1,
            "))) (. .)))"])),
    (0.5, ([(0, "the"), (1, nouns_pl), (2, "were"), (3, passive_verbs),
            (4, "by"), (5, "the"), (6, nouns), (7, ".")],
           [5, 6, 3, 0, 1, 7], "temp36",
           ["(ROOT (S (NP (DT The) (NNS ", 1, ")) (VP (VBD were) (VP (VBN ", 3,
            ") (PP (IN by) (NP (DT the) (", "nn,6",
            " ", 6, "))))) (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) (VP (VBD ", 3,
            ") (NP (DT the) (", "nn,1", " ", 1,
            "))) (. .)))"]))
]

# Subsequence: NPS
subseq_nps_templates = [(1.0, (
    [(0, "the"), (1, nouns), (2, nps_verbs), (3, "the"), (4, nouns),
     (5, "VP"), (6, ".")], [0, 1, 2, 3, 4, 6], "temp37",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (SBAR (S (NP (DT the) (", "nn,4", " ", 4, ")) ",
     "pvp,5", "))) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"]))]

# Subsequence: PP on subject
subseq_pp_on_subj_templates = [
    (1.0, ([
        (0, "the"), (1, nouns), (2, preps), (3, "the"), (4, nouns),
               (5, "VP"), (6, ".")], [
            3, 4, 5, 6], "temp38", [
                "(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ",
        "ppp,2,4", ") ", "pvp,5", " (. .)))"], [
                    "(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) ",
        "pvp,5", " (. .)))"]))]

# Subsequence: Rel clause on subject
subseq_rel_on_subj_templates = [
    (1.0, ([
        (0, "the"), (1, nouns), (2, rels), (3, transitive_verbs),
               (4, "the"), (5, nouns), (6, "VP"), (7, ".")], [
            4, 5, 6, 7], "temp39", [
                "(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ",
        "prc,2,5", ") ", "pvp,6", " (. .)))"], [
                    "(ROOT (S (NP (DT The) (", "nn,5", " ", 5, ")) ",
        "pvp,6", " (. .)))"]))]

# Subsequence: Past participles
subseq_past_participle_templates = [
    ((1.0 * len(intransitive_verbs) + len(transitive_verbs))
     / (len(intransitive_verbs) + 2 * len(transitive_verbs)), (
        [(0, "the"), (1, nouns), (2, past_participles), (3, "in"),
         (4, "the"), (5, location_nouns_b), (6, "VP"), (7, ".")],
        [0, 1, 2, 3, 4, 5, 7], "temp40",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBN ", 2,
         ") (PP (IN in) (NP (DT the) (NN ", 5, "))))) ",
         "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (PP (IN in) (NP (DT the) (NN ", 5,
         ")))) (. .)))"])),
    ((1.0 * len(transitive_verbs)) / (len(intransitive_verbs) +
                                      2 * len(transitive_verbs)), (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"),
         (4, nouns), (5, past_participles), (6, "in"),
         (7, "the"), (8, location_nouns_b), (9, ".")], [3, 4, 5, 6, 7, 8, 9],
        "temp41",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4,
         ")) (VP (VBN ", 5, ") (PP (IN in) (NP (DT the) (NN ", 8,
         ")))))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 5,
         ") (PP (IN in) (NP (DT the) (NN ", 8,
         ")))) (. .)))"]))
]

# Subsequence: NP/Z
subseq_npz_templates = [
    ((1.0 * len(npz_verbs)) / (len(npz_verbs) + len(npz_verbs_plural)), (
        [(0, conjs), (1, "the"), (2, nouns), (3, npz_verbs), (4, "the"),
         (5, nouns), (6, "VP"), (7, ".")],
        [1, 2, 3, 4, 5, 7], "temp42",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) (VP (VBD ", 3, ")))) (NP (DT the) (",
         "nn,5", " ", 5, ")) ", "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) (VP (VBD ", 3,
         ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"])),
    ((1.0 * len(npz_verbs_plural)) / (len(npz_verbs) + len(npz_verbs_plural)), (
        [(0, conjs), (1, "the"), (2, nouns_pl), (3, npz_verbs_plural),
         (4, "the"), (5, nouns), (6, "VP"), (7, ".")],
        [1, 2, 3, 4, 5, 7], "temp43",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) (VP (VBD ", 3, ")))) (NP (DT the) (",
         "nn,5", " ", 5, ")) ", "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) (VP (VBD ", 3,
         ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"]))
]

# Subsequence: Conjoined subject
subseq_conj_templates = [
    ((1.0 * len(intransitive_verbs) + len(transitive_verbs))
     / (len(intransitive_verbs) + 2 * len(transitive_verbs)), (
        [(0, "the"), (1, nouns), (2, "and"), (3, "the"), (4, nouns),
         (5, "VP"), (6, ".")], [3, 4, 5, 6], "temp44",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1,
         ")) (CC and) (NP (DT the) (", "nn,4", " ", 4, "))) ", "pvp,5",
         " (. .)))"], ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) ",
                       "pvp,5", " (. .)))"])),
    ((1.0 * len(transitive_verbs)) / (len(intransitive_verbs) +
                                      2 * len(transitive_verbs)), (
        [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"), (4, nouns),
         (5, "and"), (6, "the"), (7, nouns),
         (8, ".")], [0, 1, 2, 3, 4, 8], "temp45",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4,
         ")) (CC and) (NP (DT the) (", "nn,7", " ", 7, ")))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"]))
]

# Subsequence: Modified plural subject
subseq_adj_templates = [
    (1.0, ([
        (0, adjs), (1, nouns_pl), (2, "VP"), (3, ".")], [
            1, 2, 3], "temp46", [
                "(ROOT (S (NP (JJ ", "cap,0", ") (NNS ", 1, ")) ", "pvp,2",
        " (. .)))"], [
                    "(ROOT (S (NP (NNS ", "cap,1", ")) ",
                    "pvp,2", " (. .)))"]))]

# Subsequence: Adverb
subseq_adv_templates = [
    (1.0, ([(0, "the"), (1, nouns), (2, "VP"), (3, advs), (4, ".")], [0, 1, 2]))]

# Subsequence: Understood argument
subseq_understood_templates = [(1.0, (
    [(0, "the"), (1, nouns), (2, understood_argument_verbs), (3, "the"),
     (4, "vobj:2"), (5, ".")], [0, 1, 2, 5], "temp47",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2, ")) (. .)))"]))]

# Subsequence: Relative clause
subseq_rel_on_obj_templates = [(1.0, (
    [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"), (4, nouns),
     (5, "RC"), (6, ".")], [0, 1, 2, 3, 4, 6],
    "temp48",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ", "prc,5,5",
     ")) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"]))]

# Subsequence: PP
subseq_pp_on_obj_templates = [(1.0, (
    [(0, "the"), (1, nouns), (2, transitive_verbs), (3, "the"), (4, nouns),
     (5, preps), (6, "the"), (7, nouns), (8, ".")],
    [0, 1, 2, 3, 4, 8], "temp49",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ", "ppp,5,7",
     ")) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"]))]

# Constituent: If
const_under_if_templates = [
    (0.5, (
        [(0, advs_embed_not_entailed), (1, "the"), (2, nouns), (3, "VP"),
         (4, ","), (5, "the"), (6, nouns), (7, "VP"),
         (8, ".")], [1, 2, 3, 8], "temp50",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3",
         " (. .)))"])),
    (0.25, (
        [(0, "whether or not"), (1, "the"), (2, nouns), (3, "VP"), (4, ","),
         (5, "the"), (6, nouns), (7, "VP"), (8, ".")],
        [1, 2, 3, 8], "temp50",
        ["(ROOT (S (SBAR (IN Whether) (CC or) (RB not) (S (NP (DT the) (",
         "nn,2", " ", 2, ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3",
         " (. .)))"])),
    (0.25, ([(0, "in case"), (1, "the"), (2, nouns), (3, "VP"), (4, ","),
             (5, "the"), (6, nouns), (7, "VP"), (8, ".")],
            [1, 2, 3, 8], "temp50",
            ["(ROOT (S (SBAR (PP (IN In) (NP (NN case))) (S (NP (DT the) (",
             "nn,2", " ", 2, ")) ", "pvp,3",
             ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ",
             "pvp,7", ") (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ",
             "pvp,3", " (. .)))"]))
]

const_outside_if_templates = [(1.0, (
    [(0, advs_outside_not_entailed), (1, "the"), (2, nouns), (3, "VP"),
     (4, ","), (5, "the"), (6, nouns), (7, "VP"),
     (8, ".")], [5, 6, 7, 8], "temp51",
    ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
     ")) ", "pvp,3", ")) (, ,) (S (NP (DT the) (",
     "nn,6", " ", 6, ")) ", "pvp,7", ") (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ", "pvp,7", " (. .)))"]))]

# Constituent: Said
const_quot_templates = [
    (1.0, (
        [(0, "the"), (1, nouns), (2, nonentailing_quot_vebs), (3, "that"),
         (4, "the"), (5, nouns), (6, "VP"), (7, ".")],
        [4, 5, 6, 7], "temp52",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (SBAR (IN that) (S (NP (DT the) (", "nn,5", " ",
         5, ")) ", "pvp,6", "))) (. .)))"], ["(ROOT (S (NP (DT The) (",
                                             "nn,5", " ", 5, ")) ", "pvp,6",
                                             " (. .)))"]))
]
# All appear at least 100 with S complements "seemed", "appeared", "told",
# "reported"

# Constituent: Disjunction
const_disj_templates = [
    (0.5, (
        [(0, "the"), (1, nouns), (2, "VP"), (3, ","), (4, "or"), (5, "the"),
         (6, nouns), (7, "VP"), (8, ".")], [0, 1, 2, 8],
        "temp53",
        ["(ROOT (S (S (NP (DT The) (", "nn,1", " ", 1, ")) ", "pvp,2",
         ") (, ,) (CC or) (S (NP (DT the) (", "nn,6", " ", 6,
         ")) ", "pvp,7", ") (. .)))"], ["(ROOT (S (NP (DT The) (", "nn,1", " ",
                                        1, ")) ", "pvp,2", " (. .)))"])),
    (0.5, (
        [(0, "the"), (1, nouns), (2, "VP"), (3, ","), (4, "or"), (5, "the"),
         (6, nouns), (7, "VP"), (8, ".")], [5, 6, 7, 8],
        "temp54",
        ["(ROOT (S (S (NP (DT The) (", "nn,1", " ", 1, ")) ", "pvp,2",
         ") (, ,) (CC or) (S (NP (DT the) (", "nn,6", " ", 6,
         ")) ", "pvp,7", ") (. .)))"], ["(ROOT (S (NP (DT The) (", "nn,6", " ",
                                        6, ")) ", "pvp,7", " (. .)))"]))
]

# Constituent: Noun complements
const_noun_comp_nonent_templates = [(1.0,
                                     ([(0, "the"),
                                       (1, nouns),
                                         (2, "had"),
                                         (3, "the"),
                                         (4, nonent_complement_nouns),
                                         (5, "that"),
                                         (6, "the"),
                                         (7, nouns),
                                         (8, "VP"),
                                         (9, ".")],
                                         [6, 7, 8, 9], "temp55"))]
# All appear at least 100 times with S complements story

# Constituent: Adjective complements
const_adj_comp_nonent_templates = [(0.5, (
    [(0, "the"), (1, nouns_sg), (2, "was"), (3, adj_comp_nonent), (4, "that"),
     (5, "the"), (6, nouns), (7, "VP"), (8, ".")],
    [5, 6, 7, 8], "temp56")), (0.5, (
        [(0, "the"), (1, nouns_pl), (2, "were"), (3, adj_comp_nonent),
         (4, "that"), (5, "the"), (6, nouns), (7, "VP"),
         (8, ".")], [5, 6, 7, 8], "temp57"))]
# All appear at least 100 times with S complements

# Constituent: Probably, supposedly, ...
const_advs_nonent_templates = [(1.0, (
    [(0, advs_nonentailed), (1, "the"), (2, nouns), (3, "VP"), (4, ".")],
    [1, 2, 3, 4], "temp58",
    ["(ROOT (S (ADVP (RB ", "cap,0", ")) (S (NP (DT the) (", "nn,2", " ", 2,
     ")) ", "pvp,3", ") (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3", " (. .)))"]))]

# Constituent: Since
const_adv_embed_templates = [(1.0, (
    [(0, advs_embed_entailed), (1, "the"), (2, nouns), (3, "VP"), (4, ","),
     (5, "the"), (6, nouns), (7, "VP"), (8, ".")],
    [1, 2, 3, 8], "temp59",
    ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
     ")) ", "pvp,3", ")) (, ,) (S (NP (DT the) (",
     "nn,6", " ", 6, ")) ", "pvp,7", ") (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3", " (. .)))"]))]

const_adv_outside_templates = [
    (7.0 / 9, (
        [(0, advs_outside_entailed), (1, "the"), (2, nouns), (3, "VP"),
         (4, ","), (5, "the"), (6, nouns), (7, "VP"),
         (8, ".")], [5, 6, 7, 8], "temp60",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ", "pvp,7",
         " (. .)))"])),
    (1.0 / 9, (
        [(0, "in case"), (1, "the"), (2, nouns), (3, "VP"), (4, ","),
         (5, "the"), (6, nouns), (7, "VP"), (8, ".")],
        [5, 6, 7, 8], "temp60",
        ["(ROOT (S (SBAR (PP (IN In) (NP (NN case))) (S (NP (DT the) (",
         "nn,2", " ", 2, ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ",
         "pvp,7", " (. .)))"])),
    (1.0 / 9, (
        [(0, "whether or not"), (1, "the"), (2, nouns), (3, "VP"), (4, ","),
         (5, "the"), (6, nouns), (7, "VP"), (8, ".")],
        [5, 6, 7, 8], "temp60",
        ["(ROOT (S (SBAR (IN Whether) (CC or) (RB not) (S (NP (DT the) (",
         "nn,2", " ", 2, ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ",
         "pvp,7", ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ",
         "pvp,7", " (. .)))"]))
]

# Constituent: Knew
const_quot_ent_templates = [(1.0, (
    [(0, "the"), (1, nouns), (2, const_quot_entailed), (3, "that"),
     (4, "the"), (5, nouns), (6, "VP"), (7, ".")],
    [4, 5, 6, 7], "temp61",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (SBAR (IN that) (S (NP (DT the) (", "nn,5", " ", 5,
     ")) ", "pvp,6", "))) (. .)))"], ["(ROOT (S (NP (DT The) (",
                                      "nn,5", " ", 5, ")) ", "pvp,6",
                                      " (. .)))"]))]

# Constituent: Conjunction
const_conj_templates = [
    (0.5, ([(0, "the"), (1, nouns), (2, "VP"), (3, ","), (4, "and"),
            (5, "the"), (6, nouns), (7, "VP"), (8, ".")],
           [0, 1, 2, 8], "temp62",
           ["(ROOT (S (S (NP (DT The) (", "nn,1", " ", 1, ")) ", "pvp,2",
            ") (, ,) (CC and) (S (NP (DT the) (", "nn,6",
            " ", 6, ")) ", "pvp,7", ") (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) ", "pvp,2",
            " (. .)))"])),
    (0.5, ([(0, "the"), (1, nouns), (2, "VP"), (3, ","), (4, "and"), (5, "the"),
            (6, nouns), (7, "VP"), (8, ".")],
           [5, 6, 7, 8], "temp63",
           ["(ROOT (S (S (NP (DT The) (", "nn,1", " ", 1, ")) ", "pvp,2",
            ") (, ,) (CC and) (S (NP (DT the) (", "nn,6",
            " ", 6, ")) ", "pvp,7", ") (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ",
            "pvp,7", " (. .)))"]))
]

# Constituent: Embedded question
const_embed_quest = [(1.0, ([(0, "the"), (1, nouns),
                             (2, question_embedding_verbs), (
    3, quest), (4, "the"), (5, nouns), (6, "VP"), (7, ".")], [4, 5, 6, 7],
                            "temp64"))]

# Constituent: Noun complements
const_noun_comp_ent_templates = [(1.0, ([(0, "the"), (1, nouns), (2, "had"),
                                         (3, "the"), (
    4, ent_complement_nouns), (5, "that"), (6, "the"), (7, nouns), (8, "VP"),
                                         (9, ".")], [6, 7, 8, 9], "temp65"))]

# Constituent: Adjective complements
const_adj_comp_ent_templates = [(0.5, (
    [(0, "the"), (1, nouns_sg), (2, "was"), (3, adj_comp_ent), (4, "that"),
     (5, "the"), (6, nouns), (7, "VP"), (8, ".")],
    [5, 6, 7, 8], "temp66")), (0.5, (
        [(0, "the"), (1, nouns_pl), (2, "were"), (3, adj_comp_ent), (4, "that"),
         (5, "the"), (6, nouns), (7, "VP"), (8, ".")],
        [5, 6, 7, 8], "temp67"))]

# Constituent: Sentential adverbs
const_advs_ent_templates = [
    (5.0 / 7, ([(0, advs_entailed), (1, "the"), (2, nouns), (3, "VP"),
                (4, ".")], [1, 2, 3, 4], "temp68",
               ["(ROOT (S (ADVP (RB ", "cap,0", ")) (S (NP (DT the) (", "nn,2",
                " ", 2, ")) ", "pvp,3", ") (. .)))"],
               ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3",
                " (. .)))"])),
    (1.0 / 7, ([(0, "without a doubt"), (1, "the"), (2, nouns), (3, "VP"),
                (4, ".")], [1, 2, 3, 4], "temp68",
               ["(ROOT (S (PP (IN Without) (NP (DT a) (NN doubt))) "
                "(S (NP (DT the) (", "nn,2", " ", 2, ")) ", "pvp,3",
                ") (. .)))"], ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ",
                               "pvp,3", " (. .)))"])),
    (1.0 / 7, ([(0, "of course"), (1, "the"), (2, nouns), (3, "VP"), (4, ".")],
               [1, 2, 3, 4], "temp68",
               ["(ROOT (S (PP (IN Of) (NP (NN course))) (S (NP (DT the) (",
                "nn,2", " ", 2, ")) ", "pvp,3",
                ") (. .)))"], ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2,
                               ")) ", "pvp,3", " (. .)))"]))
]

template_list = [
    ("lexical_overlap", "ln_subject/object_swap", "non-entailment",
     lex_simple_templates),
    ("lexical_overlap", "ln_preposition", "non-entailment", lex_prep_templates),
    ("lexical_overlap", "ln_relative_clause", "non-entailment",
     lex_rc_templates),
    ("lexical_overlap", "ln_passive", "non-entailment", lex_pass_templates),
    ("lexical_overlap", "ln_conjunction", "non-entailment", lex_conj_templates),
    ("lexical_overlap", "le_relative_clause",
     "entailment", lex_rc_ent_templates),
    ("lexical_overlap", "le_around_prepositional_phrase",
     "entailment", lex_cross_pp_ent_templates),
    ("lexical_overlap", "le_around_relative_clause",
     "entailment", lex_cross_rc_ent_templates),
    ("lexical_overlap", "le_conjunction",
     "entailment", lex_ent_conj_templates),
    ("lexical_overlap", "le_passive", "entailment", lex_ent_pass_templates),
    ("subsequence", "sn_NP/S", "non-entailment", subseq_nps_templates),
    ("subsequence", "sn_PP_on_subject",
     "non-entailment", subseq_pp_on_subj_templates),
    ("subsequence", "sn_relative_clause_on_subject",
     "non-entailment", subseq_rel_on_subj_templates),
    ("subsequence", "sn_past_participle",
     "non-entailment", subseq_past_participle_templates),
    ("subsequence", "sn_NP/Z", "non-entailment", subseq_npz_templates),
    ("subsequence", "se_conjunction", "entailment", subseq_conj_templates),
    ("subsequence", "se_adjective", "entailment", subseq_adj_templates),
    ("subsequence", "se_understood_object",
     "entailment", subseq_understood_templates),
    ("subsequence", "se_relative_clause_on_obj",
     "entailment", subseq_rel_on_obj_templates),
    ("subsequence", "se_PP_on_obj", "entailment", subseq_pp_on_obj_templates),
    ("constituent", "cn_embedded_under_if",
     "non-entailment", const_under_if_templates),
    ("constituent", "cn_after_if_clause",
     "non-entailment", const_outside_if_templates),
    ("constituent", "cn_embedded_under_verb",
     "non-entailment", const_quot_templates),
    ("constituent", "cn_disjunction", "non-entailment", const_disj_templates),
    ("constituent", "cn_adverb", "non-entailment", const_advs_nonent_templates),
    ("constituent", "ce_embedded_under_since",
     "entailment", const_adv_embed_templates),
    ("constituent", "ce_after_since_clause",
     "entailment", const_adv_outside_templates),
    ("constituent", "ce_embedded_under_verb",
     "entailment", const_quot_ent_templates),
    ("constituent", "ce_conjunction", "entailment", const_conj_templates),
    ("constituent", "ce_adverb", "entailment", const_advs_ent_templates)
]

lemma = {}
lemma["professors"] = "professor"
lemma["students"] = "student"
lemma["presidents"] = "president"
lemma["judges"] = "judge"
lemma["senators"] = "senator"
lemma["secretaries"] = "secretary"
lemma["doctors"] = "doctor"
lemma["lawyers"] = "lawyer"
lemma["scientists"] = "scientist"
lemma["bankers"] = "banker"
lemma["tourists"] = "tourist"
lemma["managers"] = "manager"
lemma["artists"] = "artist"
lemma["authors"] = "author"
lemma["actors"] = "actor"
lemma["athletes"] = "athlete"
