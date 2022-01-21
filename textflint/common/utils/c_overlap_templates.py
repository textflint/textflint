import random
import numpy as np

""" Overlap templates for SMCN transformation NLIOverlap
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
        return verb+"了", "(VP (VBD " + verb + "))"
    else:
        obj = random.choice(nouns)

        return verb + " " + \
            obj, "(VP (VBD " + verb + ") (NP (DT the) (NN " + obj + ")))"
def generate_if_vp():
    verb = random.choice(verbs_if)
    if verb in intransitive_verbs:
        return verb, "(VP (VBD " + verb + "))"
    else:
        obj = random.choice(nouns)

        return  verb + \
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

    verb = random.choice(verbs_if)
    if verb in intransitive_verbs:
        return verb + " " + rel +"的"
    else:
        arg = random.choice(nouns)
        return  arg +  " " + verb + rel+"的"



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
    #print("here")
    #print(template_tuple)
    template = template_tuple[0]
    #print("template\n")
    #print(template)
    hypothesis_template = template_tuple[1]
    #print("hypothesis_template\n")
    #print(hypothesis_template)
    template_tag = template_tuple[2]
    #print("template_tag\n")
    #print(template_tag)

    premise_list = []
    index_dict = {}

    for (index, element) in template:
        #print(element)
        if element == "VP":
            vp, vp_parse = generate_vp()
            premise_list.append(vp)
            index_dict[index] = vp

        elif element == "VP_if":
            vpif, vpif_parse = generate_if_vp()
            premise_list.append(vpif)
            index_dict[index] = vpif

        elif element == "RC":
            rc = generate_rc()
            #print("rc:")
            #print(rc)
            premise_list.append(rc)
            index_dict[index] = rc


        elif "vobj" in element:
            obj = random.choice(object_dict[index_dict[int(element[-1])]])
            #print("obj")
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
    #print("premise_list：")
    #print(premise_list)
    return postprocess(" ".join(premise_list)), postprocess(
        " ".join(hypothesis_list)), template_tag


nouns_sg = [
    "教授",
    "学生",
    "总统",
    "裁判",
    "参议员",
    "秘书",
    "医生",
    "律师",
    "科学家",
    "银行工作人员",
    "游客",
    "经理",
    "艺术家",
    "作家",
    "演员",
    "运动员"]
nouns_pl = [
    "教授们",
    "学生们",
    "总统们",
    "裁判们",
    "参议员们",
    "秘书们",
    "医生们",
    "律师们",
    "科学家们",
    "银行工作人员们",
    "游客们",
    "经理们",
    "艺术家们",
    "作家们",
    "演员们",
    "运动员们"
]
nouns = nouns_sg + nouns_pl
transitive_verbs = [
    "推荐了",
    "帮助了",
    "支持",
    "联系了",
    "相信",
    "躲开了",
    "提议了",
    "看见了",
    "阻止了",
    "介绍了",
    "提到了",
    "鼓励了",
    "感谢了",
    "羡慕"]
transitive_prep_verbs = [
    "推荐",
    "帮助",
    "支持",
    "联系",
    "相信",
    "提议",
    "看见",
    "阻止",
    "介绍",
    "提到",
    "鼓励",
    "感谢",
    "羡慕"]
passive_verbs = [
    "推荐了",
    "帮助了",
    "支持了",
    "联系了",
    "相信了",
    "躲开了",
    "看见了",
    "阻止了",
    "介绍了",
    "提到了",
    "鼓励了",
    "感谢了",
    "认出了",
    ]
intransitive_verbs = [
    "睡觉",
    "跳舞",
    "跑步",
    "大叫",
    "辞职",
    "到达",
    "演出"]
verbs_if=transitive_prep_verbs+intransitive_verbs
verbs = transitive_verbs + intransitive_verbs

nps_verbs = ["相信", "知道", "听到"]
# "forgot", "preferred", "claimed", "wanted", "needed", "found", "suggested",
# "expected"] # These all appear at least 100 times with both NP and S arguments
npz_verbs = ["躲起来", "移动", "提出", "支付", "学习", "停止"]
# "paid",  "changed", "studied", "answered", "stopped", "grew", "moved",
# "returned", "left","improved", "lost", "visited", "ate", "played"]
# All appear at least 100 times with both transitive and intransitive frames
npz_verbs_plural = ["打架"]
# All appear at least 100 times with both transitive and intransitive frames
understood_argument_verbs = [
    "支付了",
    "探索了",
    "赢了",
    "写完了",
    "离开了",
    "读完了",
    "吃完了"]
nonentailing_quot_vebs = [
    "希望",
    "声称",
    "认为",
    "相信",
    "说",
    "假设"]
question_embedding_verbs = [
    "想知道",
    "理解",
    "知道",
    "问",
    "解释",
    "意识到"]

# Each appears at least 100 times in MNLI
preps = ["附近的", "后面的", "前面的", "挨着的"]
conjst = ["的时候", "之后", "之前"]
conjs = ["尽管", "因为", "自从"]
past_participles = ["学习", "支付", "帮忙", "调查", "展示"]
called_objects = ["懦夫", "骗子", "英雄", "傻瓜"]
told_objects = ["故事", "谎言", "真相", "秘密"]
food_words = [
    "水果",
    "沙拉",
    "西兰花",
    "三明治",
    "米饭",
    "玉米",
    "冰淇淋"]
location_nouns = [
    "社区",
    "地区",
    "国家",
    "小镇",
    "山谷",
    "森林",
    "花园",
    "博物馆",
    "沙漠",
    "小岛",
    "小镇"]
location_nouns_b = ["博物馆", "学校", "图书馆", "办公室", "实验室"]
won_objects = [
    "比赛",
    "比赛",
    "战争",
    "获奖",
    "竞赛",
    "选举",
    "战斗",
    "获奖",
    "锦标赛"]
read_wrote_objects = [
    "书",
    "专栏",
    "报告",
    "诗",
    "信",
    "小说",
    "故事",
    "表演",
    "演讲"]
adjs = ["重要的", "受欢迎的", "出名的", "年轻的", "开心的",
        "有帮助的", "严肃的", "生气的"]  # All at least 100 times in MNLI
adj_comp_nonent = ["恐惧的", "当然", "某个"]
adj_comp_ent = ["抱歉的", "有意识的", "高兴的"]
advs = ["快地", "慢地", "高兴地", "简单地", "安静地",
        "深思熟虑地"]  # All at least 100 times in MNLI
const_adv = [
    "之后",
    "之前",
    "因为",
    "尽管",
    "尽管",
    "自从",
    "然而"
]
const_quot_entailed = ["忘记", "了解", "记住", "知道"]
advs_nonentailed_before = ["据说", "希望"]
advs_nonentailed_after = ["大概", "可能"]
advs_entailed = ["当然", "确实", "清楚地", "显然", "突然"]
rels = ["过"]
quest = ["为什么", "怎样"]
nonent_complement_nouns = ["感觉", "证据", "想法", "信仰"]
ent_complement_nouns = ["事实", "原因", "新闻", "时间"]
object_dict = {}
object_dict["打电话"] = called_objects
object_dict["告诉"] = told_objects
object_dict["带来"] = food_words
object_dict["使"] = food_words
object_dict["拯救"] = food_words
object_dict["提供"] = food_words
object_dict["支付了"] = nouns
object_dict["探索了"] = location_nouns
object_dict["赢了"] = won_objects
object_dict["写完了"] = read_wrote_objects
object_dict["离开了"] = location_nouns
object_dict["读完了"] = read_wrote_objects
object_dict["吃完了"] = food_words

advs_embed_not_entailed = ["如果", "万一"]
advs_embed_entailed = [
    "之后",
    "之前",
    "的时候"]
advs_outside_not_entailed = ["如果", "万一"]
advs_outside_entailed_time = [
    "之后",
    "之前",
    "的时候"
    ]
advs_outside_entailed=[
    "因为",
    "尽管",
    "自从",
    ]
# Lexical Overlap: Simple sentence
lex_simple_templates = [(1.0, (
    [(0, nouns), (1, transitive_verbs),
     (2, nouns), (3, "。")], [ 2, 1,0,3], "temp1",
    ["(ROOT (S (NP (DT ) (", "nn,0", " ", 0,
     ")) (VP (VBD ", 1, ") (NP (DT ) (", "nn,2", " ", 2, "))) (. .)))"],
    ["(ROOT (S (NP (DT ) (", "nn,2", " ", 2,
     ")) (VP (VBD ", 1, ") (NP (DT ) (", "nn,0", " ", 0, "))) (. .)))"]))]

# Lexical Overlap: Preposition on subject
lex_prep_templates = [
    (1.0 / 6, (
        [(1, nouns), (2, preps),
         (4, nouns), (5, transitive_verbs), (7, nouns),
         (8, "。")], [ 4, 5,  1, 8], "temp2",
        ["(ROOT (S (NP (NP (DT ) (", "nn,1", " ", 1, ")) ",
         "ppp,4,2", ") (VP (VBD ", 5, ") (NP (DT ) (", "nn,7", " ",
         7, "))) (. .)))"],
        ["(ROOT (S (NP (DT ) (", "nn,4", " ", 4, ")) (VP (VBD ", 5,
         ") (NP (DT ) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 6, (
        [(1, nouns), (2, preps),  (4, nouns),
         (5, transitive_verbs),  (7, nouns),
         (8, "。")], [ 7, 5, 1, 8], "temp3",
        ["(ROOT (S (NP (NP (DT) (", "nn,1", " ", 1, ")) ", "ppp,4,2",
         ") (VP (VBD ", 5, ") (NP (DT) (", "nn,7", " ",
         7, "))) (. .)))"],
        ["(ROOT (S (NP (DT ) (", "nn,7", " ", 7, ")) (VP (VBD ", 5,
         ") (NP (DT) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 6, (
        [ (1, nouns), (2, preps),  (4, nouns),
         (5, transitive_verbs), (7, nouns),
         (8, "。")], [ 7, 5, 4, 8], "temp4",
        ["(ROOT (S (NP (NP (DT ) (", "nn,1", " ", 1, ")) ", "ppp,4,2",
         ") (VP (VBD ", 5, ") (NP (DT ) (", "nn,7", " ",
         7, "))) (. .)))"],
        ["(ROOT (S (NP (DT) (", "nn,7", " ", 7, ")) (VP (VBD ", 5,
         ") (NP (DT) (", "nn,4", " ", 4, "))) (. .)))"])),
    (1.0 / 6, (
        [ (1, nouns), (2, transitive_verbs),
         (4, nouns), (5, preps),  (7, nouns),
         (8, "。")], [4, 2, 1, 8], "temp5",
        ["(ROOT (S (NP (DT) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT) (", "nn,4", " ", 4, ")) ",
         "ppp,7,5", ")) (. .)))"],
        ["(ROOT (S (NP (DT ) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
         ") (NP (DT) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 6, (
        [ (1, nouns), (2, transitive_verbs),
         (4, nouns), (5, preps), (7, nouns),
         (8, "。")], [4, 2, 7, 8], "temp6",
        ["(ROOT (S (NP (DT) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT) (", "nn,4", " ", 4, ")) ",
         "ppp,7,5", ")) (. .)))"],
        ["(ROOT (S (NP (DT) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
         ") (NP (DT) (", "nn,7", " ", 7, "))) (. .)))"])),
    (1.0 / 6, (
        [ (1, nouns), (2, transitive_verbs),
         (4, nouns), (5, preps),  (7, nouns),
         (8, "。")], [7, 2, 1, 8], "temp7",
        ["(ROOT (S (NP (DT) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT) (", "nn,4", " ", 4, ")) ",
         "ppp,7,5", ")) (. .)))"],
        ["(ROOT (S (NP (DT) (", "nn,7", " ", 7, ")) (VP (VBD ", 2,
         ") (NP (DT) (", "nn,1", " ", 1, "))) (. .)))"]))
]

# Lexical Overlap: Relative clause on subject
lex_rc_templates = [
    (1.0 / 6, (
        [ (1, nouns), (2, transitive_prep_verbs),(3, rels), (4,"的"), (5, nouns),
          (6, transitive_verbs),
          (8, nouns), (9, "。")], [ 5, 6,  1, 9], "temp8",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,5,2",
         ") (VP (VBD ", 6, ") (NP (DT the) (", "nn,8", " ",
         8, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 6,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 6, (
        [ (1, nouns), (2, transitive_prep_verbs), (3, rels), (4,"的"), (5, nouns), (6, transitive_verbs),
         (8, nouns), (9, "。")], [ 8, 6,1, 9], "temp9",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,5,2",
         ") (VP (VBD ", 6, ") (NP (DT the) (", "nn,8", " ",
         8, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,8", " ", 8, ")) (VP (VBD ", 6,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 6, (
        [ (1, nouns), (2, transitive_prep_verbs),(3, rels), (4,"的"), (5, nouns), (6, transitive_verbs),
          (8, nouns), (9, "。")], [ 8, 6, 5, 9], "temp10",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,5,2",
         ") (VP (VBD ", 6, ") (NP (DT the) (", "nn,8", " ",
         8, "))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,8", " ", 8, ")) (VP (VBD ", 6,
         ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"])),

    (1.0 / 6, (
        [(1, nouns), (2, transitive_verbs),(4, nouns),
          (5, transitive_prep_verbs),(6, rels),(7,"的"),
          (8, nouns), (9, "。")], [ 4, 2, 1, 9], "temp14",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "prc,5,8", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"])),
    (1.0 / 6, (
        [(1, nouns), (2, transitive_verbs), (4, nouns),
         (5, transitive_prep_verbs), (6, rels),(7,"的"),
          (8, nouns), (9, "。")], [4, 2,  8, 9], "temp15",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "prc,5,8", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,8", " ", 8, "))) (. .)))"])),
    (1.0 / 6, (
        [(1, nouns), (2, transitive_verbs),  (4, nouns),
         (5, transitive_prep_verbs), (6, rels),(7,"的"),
          (8, nouns), (9, "。")], [ 8, 2, 1, 9], "temp16",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ",
         "prc,5,8", ")) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,8", " ", 8, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,1", " ", 1, "))) (. .)))"]))
]

# Lexical Overlap: Passive incorrect
lex_pass_templates = [
    (1.0, ([ (1, nouns), (2, "被"),(3, nouns), (4, passive_verbs),
              (7, "。")],
           [ 1, 4, 3, 7], "temp20",
           ["(ROOT (S (NP (DT The) (NN ", 1, ")) (VP (VBD was) (VP (VBN ", 3,
            ") (PP (IN by) (NP (DT the) (", "nn,6",
            " ", 6, "))))) (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 3,
            ") (NP (DT the) (", "nn,6", " ", 6,
            "))) (. .)))"]))
]

# Lexical Overlap: Conjunctions
lex_conj_templates = [
    (0.25, ([ (1, nouns), (2, "和"), (4, nouns),
             (5, transitive_verbs),  (7, nouns),
             (8, "。")], [1, 5,  4, 8], "temp22",
            ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1,
             ")) (CC and) (NP (DT the) (", "nn,4", " ", 4,
             "))) (VP (VBD ", 5, ") (NP (DT the) (", "nn,7",
             " ", 7, "))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 5,
             ") (NP (DT the) (", "nn,4", " ", 4,
             "))) (. .)))"])),
    (0.25, ([ (1, nouns), (2, "和"),  (4, nouns),
             (5, transitive_verbs),  (7, nouns),
             (8, "。")], [ 4, 5, 1, 8], "temp23",
            ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1,
             ")) (CC and) (NP (DT the) (", "nn,4", " ", 4,
             "))) (VP (VBD ", 5, ") (NP (DT the) (", "nn,7", " ", 7,
             "))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 5,
             ") (NP (DT the) (", "nn,1", " ", 1,
             "))) (. .)))"])),
    (0.25, ([ (1, nouns), (2, transitive_verbs),
             (4, nouns), (5, "和"),  (7, nouns),
             (8, "。")], [4, 2, 7, 8], "temp24",
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
             ") (NP (NP (DT the) (", "nn,4", " ", 4,
             ")) (CC and) (NP (DT the) (", "nn,7", " ", 7, ")))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 2,
             ") (NP (DT the) (", "nn,7", " ", 7,
             "))) (. .)))"])),
    (0.25, ([ (1, nouns), (2, transitive_verbs),
             (4, nouns), (5, "和"), (7, nouns),
             (8, "。")], [ 7, 2,  4, 8], "temp25",
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
             ") (NP (NP (DT the) (", "nn,4", " ", 4,
             ")) (CC and) (NP (DT the) (", "nn,7", " ", 7, ")))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,7", " ", 7, ")) (VP (VBD ", 2,
             ") (NP (DT the) (", "nn,4", " ", 4,
             "))) (. .)))"]))
]

# Lexical Overlap: Relative clause
lex_rc_ent_templates = [
    (0.25, ([ (1, nouns), (5, transitive_prep_verbs),(2, rels),  (4, nouns),
             (6, transitive_verbs),
              (8, nouns), (9, "。")], [4, 5, 1, 9], "temp26",
            ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,5",
             ") (VP (VBD ", 6, ") (NP (DT the) (",
             "nn,8", " ", 8, "))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 5,
             ") (NP (DT the) (", "nn,1", " ", 1,
             "))) (. .)))"])),
    (0.25, ([(3, transitive_prep_verbs),(2, rels), (5, nouns),(0, "的"), (1, nouns),
              (6, transitive_verbs),
              (8, nouns), (9, "。")], [ 1, 3,  5, 9], "temp27",
            ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,5",
             ") (VP (VBD ", 6, ") (NP (DT the) (",
             "nn,8", " ", 8, "))) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 3,
             ") (NP (DT the) (", "nn,5", " ", 5,
             "))) (. .)))"])),
    (0.25, ([(1, nouns), (2, transitive_verbs), (6, transitive_prep_verbs),(5, rels),(8, nouns), (3, "的"),
             (4, nouns), (9, "。")], [ 4, 6,  8, 9], "temp28",
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
             ") (NP (NP (DT the) (", "nn,4", " ", 4,
             ")) ", "prc,5,8", ")) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) (VP (VBD ", 6,
             ") (NP (DT the) (", "nn,8", " ", 8,
             "))) (. .)))"])),
    (0.25, ([(1, nouns), (2, transitive_verbs), (7, nouns),(8, transitive_prep_verbs),(5, rels), (3, "的"),
             (4, nouns),
              (9, "。")], [7, 8,  4, 9], "temp29",
            ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
             ") (NP (NP (DT the) (", "nn,4", " ", 4,
             ")) ", "prc,5,8", ")) (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,7", " ", 7, ")) (VP (VBD ", 8,
             ") (NP (DT the) (", "nn,4", " ", 4,
             "))) (. .)))"]))
]

# Lexical Overlap: Across PP
lex_cross_pp_ent_templates = [(1.0, (
    [ (4, nouns), (2, preps),  (1, nouns),
     (5, transitive_verbs),  (7, nouns), (8, "。")],
    [ 1, 5,  7, 8], "temp30",
    ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "ppp,2,4",
     ") (VP (VBD ", 5, ") (NP (DT the) (", "nn,7", " ", 7,
     "))) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 5,
     ") (NP (DT the) (", "nn,7", " ", 7, "))) (. .)))"]))]

# Lexical Overlap: Across relative clause
lex_cross_rc_ent_templates = [
    (1.0, (
    [(0, intransitive_verbs),(1, "的"), (2, nouns),  (3, transitive_verbs),
     (5, nouns), (6, "。")], [2, 3,  5, 6],
    "temp31",
    ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ", "prc,2,2",
     ") (VP (VBD ", 3, ") (NP (DT the) (", "nn,5", " ", 5,
     "))) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 3,
     ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"])),

    ]

# Lexical Overlap: Conjunctions
lex_ent_conj_templates = [
    (0.5, ([(1, nouns), (2, "和"),  (4, nouns),
            (5, transitive_verbs),  (7, nouns),
            (8, "。")], [ 1, 5,  7, 8], "temp32",
           ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1,
            ")) (CC and) (NP (DT the) (", "nn,4", " ", 4,
            "))) (VP (VBD ", 5, ") (NP (DT the) (", "nn,7", " ", 7,
            "))) (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 5,
            ") (NP (DT the) (", "nn,7", " ", 7,
            "))) (. .)))"])),
    (0.5, ([ (1, nouns), (2, transitive_verbs),
            (4, nouns), (5, "和"), (7, nouns),
            (8, "。")], [1, 2,  7, 8], "temp33",
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
         (1, nouns), (2, advs), (3, intransitive_verbs), (4, "。")], [
           1, 3, 4], "temp34"))]

# Lexical Overlap: Passive
lex_ent_pass_templates = [
    (1.0, ([ (1, nouns), (2, "被"),(6, nouns), (3, passive_verbs),
              (7, "。")],
           [ 6, 3,  1, 7], "temp35",
           ["(ROOT (S (NP (DT The) (NN ", 1, ")) (VP (VBD was) (VP (VBN ", 3,
            ") (PP (IN by) (NP (DT the) (", "nn,6",
            " ", 6, "))))) (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) (VP (VBD ", 3,
            ") (NP (DT the) (", "nn,1", " ", 1,
            "))) (. .)))"])),

]

# Subsequence: NPS
subseq_nps_templates = [(1.0, (
    [ (1, nouns), (2, nps_verbs),  (4, nouns),
     (5, "VP"), (6, "。")], [1, 2,  4, 6], "temp37",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (SBAR (S (NP (DT the) (", "nn,4", " ", 4, ")) ",
     "pvp,5", "))) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"]))]

# Subsequence: PP on subject
subseq_pp_on_subj_templates = [
    (1.0, ([
         (1, nouns), (2, preps),  (4, nouns),
               (5, "VP"), (6, "。")], [
             4, 5, 6], "temp38", [
                "(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ",
        "ppp,2,4", ") ", "pvp,5", " (. .)))"], [
                    "(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) ",
        "pvp,5", " (. .)))"]))]

# Subsequence: Rel clause on subject
subseq_rel_on_subj_templates = [
    (1.0, ([
         (1, nouns),  (3, transitive_prep_verbs),(2, rels),(4,"的"),
                (5, nouns), (6, "VP"), (7, "。")], [
             5, 6, 7], "temp39", [
                "(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) ",
        "prc,2,5", ") ", "pvp,6", " (. .)))"], [
                    "(ROOT (S (NP (DT The) (", "nn,5", " ", 5, ")) ",
        "pvp,6", " (. .)))"]))]

# Subsequence: Past participles
subseq_past_participle_templates = [
    ((1.0 * len(intransitive_verbs) + len(transitive_verbs))
     / ((len(intransitive_verbs) + 2 * len(transitive_verbs))), (
        [(0, "在"),(1, location_nouns_b), (2, past_participles),(3, "的"),(4, nouns),
         (5, "VP"), (6, "。")],
        [ 4, 0,1,2, 6], "temp40",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBN ", 2,
         ") (PP (IN in) (NP (DT the) (NN ", 5, "))))) ",
         "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (PP (IN in) (NP (DT the) (NN ", 5,
         ")))) (. .)))"])),

    ((1.0 * len(transitive_verbs)) / (len(intransitive_verbs) +
                                      2 * len(transitive_verbs)), (
        [(1, nouns), (2, transitive_verbs),(3, nouns),
         (4, "在"),(5, location_nouns_b),(6, past_participles),
           (7, "。")], [ 1, 4, 5, 6, 7],
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
    ((1.0 * len(npz_verbs)) / ((len(npz_verbs) + len(npz_verbs_plural))*2.0), (
        [(1, "在"),  (2, nouns), (3, npz_verbs),(0, conjst),
         (5, nouns), (6, "VP"), (7, "。")],
        [2, 6,7], "temp42.1",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) (VP (VBD ", 3, ")))) (NP (DT the) (",
         "nn,5", " ", 5, ")) ", "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) (VP (VBD ", 3,
         ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"])),
    ((1.0 * len(npz_verbs)) / (((len(npz_verbs) + len(npz_verbs_plural))*2.0)*3.0), (
        [(0, "因为"),  (2, nouns), (3, npz_verbs),(1, "了"),(4, "所以"),
         (5, nouns), (6, "VP"), (7, "。")],
        [2, 6 ,7], "temp42.2",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) (VP (VBD ", 3, ")))) (NP (DT the) (",
         "nn,5", " ", 5, ")) ", "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) (VP (VBD ", 3,
         ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"])),
    ((1.0 * len(npz_verbs)) / (((len(npz_verbs) + len(npz_verbs_plural))*2.0)*3.0), (
        [(0, "尽管"),  (2, nouns), (3, npz_verbs),(1, "了"),
         (5, nouns),(4, "还是"), (6, "VP"), (7, "。")],
        [2, 6 ,7], "temp42.2",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) (VP (VBD ", 3, ")))) (NP (DT the) (",
         "nn,5", " ", 5, ")) ", "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) (VP (VBD ", 3,
         ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"])),
    ((1.0 * len(npz_verbs)) / (((len(npz_verbs) + len(npz_verbs_plural))*2.0)*3.0), (
        [(0, "自从"),  (2, nouns), (3, npz_verbs),(1, "了"),
         (5, nouns), (4, "就"),(6, "VP"), (7, "。")],
        [2, 6 ,7], "temp42.2",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) (VP (VBD ", 3, ")))) (NP (DT the) (",
         "nn,5", " ", 5, ")) ", "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) (VP (VBD ", 3,
         ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"])),
    ((1.0 * len(npz_verbs_plural)) / (((len(npz_verbs) + len(npz_verbs_plural))*2.0)*3.0), (
        [(0, "因为"), (2, nouns_pl), (3, npz_verbs_plural),(1, "了"),(4, "所以"),
          (5, nouns), (6, "VP"), (7, "。")],
        [2, 6, 7], "temp43.1",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) (VP (VBD ", 3, ")))) (NP (DT the) (",
         "nn,5", " ", 5, ")) ", "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) (VP (VBD ", 3,
         ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"])),
    ((1.0 * len(npz_verbs_plural)) / (((len(npz_verbs) + len(npz_verbs_plural))*2.0)*3.0), (
        [(0, "尽管"), (2, nouns_pl), (3, npz_verbs_plural),(1, "了"),
          (5, nouns),(4, "还是"), (6, "VP"), (7, "。")],
        [2, 6, 7], "temp43.1",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) (VP (VBD ", 3, ")))) (NP (DT the) (",
         "nn,5", " ", 5, ")) ", "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) (VP (VBD ", 3,
         ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"])),
((1.0 * len(npz_verbs_plural)) / (((len(npz_verbs) + len(npz_verbs_plural))*2.0)*3.0), (
        [(0, "自从"), (2, nouns_pl), (3, npz_verbs_plural),(1, "了"),
          (5, nouns),(4, "就"), (6, "VP"), (7, "。")],
        [2, 6, 7], "temp43.1",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) (VP (VBD ", 3, ")))) (NP (DT the) (",
         "nn,5", " ", 5, ")) ", "pvp,6", " (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) (VP (VBD ", 3,
         ") (NP (DT the) (", "nn,5", " ", 5, "))) (. .)))"])),
    ((1.0 * len(npz_verbs_plural)) / ((len(npz_verbs) + len(npz_verbs_plural))*2.0), (
        [(1, "在"), (2, nouns_pl), (3, npz_verbs_plural),(0, conjst),
          (5, nouns), (6, "VP"), (7, "。")],
        [2, 6, 7], "temp43.2",
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
        [(1, nouns), (2, "和"), (4, nouns),
         (5, "VP"), (6, "。")], [ 4, 5, 6], "temp44",
        ["(ROOT (S (NP (NP (DT The) (", "nn,1", " ", 1,
         ")) (CC and) (NP (DT the) (", "nn,4", " ", 4, "))) ", "pvp,5",
         " (. .)))"], ["(ROOT (S (NP (DT The) (", "nn,4", " ", 4, ")) ",
                       "pvp,5", " (. .)))"])),
    ((1.0 * len(transitive_verbs)) / (len(intransitive_verbs) +
                                      2 * len(transitive_verbs)), (
        [ (1, nouns), (2, transitive_verbs),(4, nouns),
         (5, "和"),  (7, nouns),
         (8, "。")], [ 1, 2,  4, 8], "temp45",
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (NP (DT the) (", "nn,4", " ", 4,
         ")) (CC and) (NP (DT the) (", "nn,7", " ", 7, ")))) (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
         ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"]))
]

# Subsequence: Modified plural subject
subseq_adj_templates = [
    (1.0, ([
        (0, adjs), (1, nouns_pl), (2, "VP"), (3, "。")], [
            1, 2, 3], "temp46", [
                "(ROOT (S (NP (JJ ", "cap,0", ") (NNS ", 1, ")) ", "pvp,2",
        " (. .)))"], [
                    "(ROOT (S (NP (NNS ", "cap,1", ")) ",
                    "pvp,2", " (. .)))"]))]

# Subsequence: Adverb
subseq_adv_templates = [
    (1.0, ([(0, "the"), (1, nouns), (2, "VP"), (3, advs), (4, "。")], [0, 1, 2]))]

# Subsequence: Understood argument
subseq_understood_templates = [(1.0, (
    [ (1, nouns), (2, understood_argument_verbs),
     (4, "vobj:2"), (5, "。")], [1, 2, 5], "temp47",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2, ")) (. .)))"]))]

# Subsequence: Relative clause
subseq_rel_on_obj_templates = [(1.0, (
    [(1, nouns), (2, transitive_verbs), (5, "RC"),(4, nouns),
      (6, "。")], [ 1, 2,  4, 6],
    "temp48",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ", "prc,5,5",
     ")) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"]))]

# Subsequence: PP
subseq_pp_on_obj_templates = [(1.0, (
    [(1, nouns), (2, transitive_verbs),  (7, nouns),
     (5, preps), (4, nouns), (8, "。")],
    [1, 2,  4, 8], "temp49",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (NP (DT the) (", "nn,4", " ", 4, ")) ", "ppp,5,7",
     ")) (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (NP (DT the) (", "nn,4", " ", 4, "))) (. .)))"]))]

# Constituent: If
const_under_if_templates = [
    (0.5, (
        [(0, advs_embed_not_entailed),  (2, nouns), (3, "VP"),
         (4, ","),  (6, nouns), (1, "就"),(7, "VP_if"),
         (8, "。")], [ 2, 3, 8], "temp50",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3",
         " (. .)))"])),
    (0.25, (
        [(0, "无论"),  (2, nouns), (3, "VP"),(1, "没有"), (4, ","),
          (6, nouns), (5, "都"),(7, "VP_if"), (8, "。")],
        [2, 3, 8], "temp50",
        ["(ROOT (S (SBAR (IN Whether) (CC or) (RB not) (S (NP (DT the) (",
         "nn,2", " ", 2, ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3",
         " (. .)))"])),
    (0.25, ([(0, "除非"),  (2, nouns), (3, "VP"), (4, ","),
              (6, nouns), (5, "才"),(7, "VP_if"), (8, "。")],
            [ 2, 3, 8], "temp50",
            ["(ROOT (S (SBAR (PP (IN In) (NP (NN case))) (S (NP (DT the) (",
             "nn,2", " ", 2, ")) ", "pvp,3",
             ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ",
             "pvp,7", ") (. .)))"],
            ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ",
             "pvp,3", " (. .)))"]))
]

const_outside_if_templates = [(1.0, (
    [(0, advs_outside_not_entailed),  (2, nouns), (3, "VP"),
     (4, ","),  (6, nouns), (5, "就"),(7, "VP_if"),
     (8, "。")], [ 6, 7, 8], "temp51",
    ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
     ")) ", "pvp,3", ")) (, ,) (S (NP (DT the) (",
     "nn,6", " ", 6, ")) ", "pvp,7", ") (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ", "pvp,7", " (. .)))"]))]

# Constituent: Said
const_quot_templates = [
    (1.0, (
        [ (1, nouns), (2, nonentailing_quot_vebs),
         (5, nouns), (6, "VP"), (7, "。")],
        [ 5, 6, 7], "temp52",
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
        [(0, "要么"), (1, nouns), (2, "VP"), (3, ","), (4, "要么"),
         (6, nouns), (7, "VP"), (8, "。")], [ 1, 2, 8],
        "temp53",
        ["(ROOT (S (S (NP (DT The) (", "nn,1", " ", 1, ")) ", "pvp,2",
         ") (, ,) (CC or) (S (NP (DT the) (", "nn,6", " ", 6,
         ")) ", "pvp,7", ") (. .)))"], ["(ROOT (S (NP (DT The) (", "nn,1", " ",
                                        1, ")) ", "pvp,2", " (. .)))"])),
    (0.5, (
        [(0, "要么"), (1, nouns), (2, "VP"), (3, ","), (4, "要么"),
         (6, nouns), (7, "VP"), (8, "。")], [6, 7, 8],
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
                                         (9, "。")],
                                         [6, 7, 8, 9], "temp55"))]
# All appear at least 100 times with S complements story

# Constituent: Adjective complements
const_adj_comp_nonent_templates = [(0.5, (
    [(0, "the"), (1, nouns_sg), (2, "was"), (3, adj_comp_nonent), (4, "that"),
     (5, "the"), (6, nouns), (7, "VP"), (8, "。")],
    [5, 6, 7, 8], "temp56")), (0.5, (
        [(0, "the"), (1, nouns_pl), (2, "were"), (3, adj_comp_nonent),
         (4, "that"), (5, "the"), (6, nouns), (7, "VP"),
         (8, ".")], [5, 6, 7, 8], "temp57"))]
# All appear at least 100 times with S complements

# Constituent: Probably, supposedly, ...
const_advs_nonent_templates = [(0.5, (
    [ (2, nouns), (0, advs_nonentailed_after), (3, "VP"), (4, "。")],
    [2, 3, 4], "temp58",
    ["(ROOT (S (ADVP (RB ", "cap,0", ")) (S (NP (DT the) (", "nn,2", " ", 2,
     ")) ", "pvp,3", ") (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3", " (. .)))"])),
    (0.5, (
    [(0, advs_nonentailed_before),  (2, nouns), (3, "VP"), (4, "。")],
    [2, 3, 4], "temp58",
    ["(ROOT (S (ADVP (RB ", "cap,0", ")) (S (NP (DT the) (", "nn,2", " ", 2,
     ")) ", "pvp,3", ") (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3", " (. .)))"]))
    ]

# Constituent: Since
const_adv_embed_templates = [(1.0, (
    [(1, "在"), (2, nouns),  (3, "VP"),(0, advs_embed_entailed), (4, ","),
     (6, nouns), (7, "VP"), (8, "。")],
    [2, 3, 8], "temp59",
    ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
     ")) ", "pvp,3", ")) (, ,) (S (NP (DT the) (",
     "nn,6", " ", 6, ")) ", "pvp,7", ") (. .)))"],
    ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3", " (. .)))"]))]

const_adv_outside_templates = [
    (1.0 / 9, (
        [(0, "尽管"),  (2, nouns), (3, "VP"),
         (4, ","), (6, nouns), (5, "还是"), (7, "VP"),
         (8, "。")], [6, 7, 8], "temp60",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ", "pvp,7",
         " (. .)))"])),
    (1.0 / 9, (
        [(0, "因为"),  (2, nouns), (3, "VP"),
         (4, ","), (5, "所以"), (6, nouns), (7, "VP"),
         (8, "。")], [ 6, 7, 8], "temp60",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ", "pvp,7",
         " (. .)))"])),
    (1.0 / 9, (
        [(0, "自从"),  (2, nouns), (3, "VP"),
         (4, ","),  (6, nouns),(5, "就"), (7, "VP"),
         (8, "。")], [ 6, 7, 8], "temp60",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ", "pvp,7",
         " (. .)))"])),
    (4.0 / 9, (
        [(1, "在"), (2, nouns), (3, "VP_if"),(0, advs_outside_entailed_time),
         (4, ","), (6, nouns),  (7, "VP"),
         (8, "。")], [ 6, 7, 8], "temp60",
        ["(ROOT (S (SBAR (IN ", "cap,0", ") (S (NP (DT the) (", "nn,2", " ", 2,
         ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ", "pvp,7",
         " (. .)))"])),
    (1.0 / 9, (
        [(0, "万一"), (2, nouns), (3, "VP"), (4, ","),
          (6, nouns), (5, "就"), (7, "VP_if"), (8, "。")],
        [6, 7, 8], "temp60",
        ["(ROOT (S (SBAR (PP (IN In) (NP (NN case))) (S (NP (DT the) (",
         "nn,2", " ", 2, ")) ", "pvp,3",
         ")) (, ,) (S (NP (DT the) (", "nn,6", " ", 6, ")) ", "pvp,7",
         ") (. .)))"],
        ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ",
         "pvp,7", " (. .)))"])),
    (1.0 / 9, (
        [(0, "无论"),  (2, nouns), (3, "VP_if"),(1, "没有"), (4, ","),
          (6, nouns),(5, "都会"), (7, "VP_if"), (8, "。")],
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
    [ (1, nouns), (2, const_quot_entailed),
      (5, nouns), (4, "已经"),(6, "VP"), (7, "。")],
    [ 5, 6, 7], "temp61",
    ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) (VP (VBD ", 2,
     ") (SBAR (IN that) (S (NP (DT the) (", "nn,5", " ", 5,
     ")) ", "pvp,6", "))) (. .)))"], ["(ROOT (S (NP (DT The) (",
                                      "nn,5", " ", 5, ")) ", "pvp,6",
                                      " (. .)))"]))]

# Constituent: Conjunction
const_conj_templates = [
    (0.5, ([ (1, nouns), (2, "VP"), (3, ","), (4, "并且"),
             (6, nouns), (7, "VP"), (8, "。")],
           [ 1, 2, 8], "temp62",
           ["(ROOT (S (S (NP (DT The) (", "nn,1", " ", 1, ")) ", "pvp,2",
            ") (, ,) (CC and) (S (NP (DT the) (", "nn,6",
            " ", 6, ")) ", "pvp,7", ") (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,1", " ", 1, ")) ", "pvp,2",
            " (. .)))"])),
    (0.5, ([ (1, nouns), (2, "VP"), (3, ","), (4, "并且"),
            (6, nouns), (7, "VP"), (8, "。")],
           [6, 7, 8], "temp63",
           ["(ROOT (S (S (NP (DT The) (", "nn,1", " ", 1, ")) ", "pvp,2",
            ") (, ,) (CC and) (S (NP (DT the) (", "nn,6",
            " ", 6, ")) ", "pvp,7", ") (. .)))"],
           ["(ROOT (S (NP (DT The) (", "nn,6", " ", 6, ")) ",
            "pvp,7", " (. .)))"]))
]

# Constituent: Embedded question
const_embed_quest = [(1.0, ([(0, "the"), (1, nouns),
                             (2, question_embedding_verbs), (
    3, quest), (4, "the"), (5, nouns), (6, "VP"), (7, "。")], [4, 5, 6, 7],
                            "temp64"))]

# Constituent: Noun complements
const_noun_comp_ent_templates = [(1.0, ([(0, "the"), (1, nouns), (2, "had"),
                                         (3, "the"), (
    4, ent_complement_nouns), (5, "that"), (6, "the"), (7, nouns), (8, "VP"),
                                         (9, "。")], [6, 7, 8, 9], "temp65"))]

# Constituent: Adjective complements
const_adj_comp_ent_templates = [(0.5, (
    [(0, "the"), (1, nouns_sg), (2, "was"), (3, adj_comp_ent), (4, "that"),
     (5, "the"), (6, nouns), (7, "VP"), (8, "。")],
    [5, 6, 7, 8], "temp66")), (0.5, (
        [(0, "the"), (1, nouns_pl), (2, "were"), (3, adj_comp_ent), (4, "that"),
         (5, "the"), (6, nouns), (7, "VP"), (8, "。")],
        [5, 6, 7, 8], "temp67"))]

# Constituent: Sentential adverbs
const_advs_ent_templates = [
    (5.0 / 7, ([ (2, nouns),(0, advs_entailed), (3, "VP"),
                (4, "。")], [ 2, 3, 4], "temp68",
               ["(ROOT (S (ADVP (RB ", "cap,0", ")) (S (NP (DT the) (", "nn,2",
                " ", 2, ")) ", "pvp,3", ") (. .)))"],
               ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ", "pvp,3",
                " (. .)))"])),
    (1.0 / 7, ([(0, "毫无疑问") , (2, nouns), (3, "VP"),
                (4, "。")], [ 2, 3, 4], "temp68",
               ["(ROOT (S (PP (IN Without) (NP (DT a) (NN doubt))) "
                "(S (NP (DT the) (", "nn,2", " ", 2, ")) ", "pvp,3",
                ") (. .)))"], ["(ROOT (S (NP (DT The) (", "nn,2", " ", 2, ")) ",
                               "pvp,3", " (. .)))"])),
    (1.0 / 7, ([ (2, nouns),(0, "当然"), (3, "VP"), (4, "。")],
               [2, 3, 4], "temp68",
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
lemma["教授们"] = "教授"
lemma["学生们"] = "学生"
lemma["总统们"] = "总统"
lemma["裁判们"] = "裁判"
lemma["参议员们"] = "参议员"
lemma["秘书们"] = "秘书"
lemma["医生们"] = "医生"
lemma["律师们"] = "律师"
lemma["科学家们"] = "科学家"
lemma["银行工作人员们"] = "银行工作人员"
lemma["游客们"] = "游客"
lemma["经理们"] = "经理"
lemma["艺术家们"] = "艺术家"
lemma["作家们"] = "作家"
lemma["演员们"] = "演员"
lemma["运动员们"] = "运动员"
