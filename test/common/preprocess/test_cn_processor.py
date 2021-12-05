import unittest

from textflint.common.preprocess.cn_processor import CnProcessor

sents = [
    "我们正在玩乒乓球，你真的太懒了。她打的很漂亮。",
    '我不理解这个问题！',
    "2021年计划我们写了三张纸。"
]


class TestCnProcessor(unittest.TestCase):
    test_processor = CnProcessor()

    def test_tokenize(self):
        self.assertRaises(AssertionError, self.test_processor.tokenize, [])
        self.assertEqual([], self.test_processor.tokenize(''))
        self.assertEqual(
            ['我', '们', '正', '在', '玩', '乒', '乓', '球', '，', '你', '真', '的', '太', '懒', '了', '。', '她', '打', '的', '很',
             '漂', '亮', '。'], self.test_processor.tokenize(
                sents[0], cws=False))
        self.assertEqual(
            ['我们', '正在', '玩', '乒乓球', '，', '你', '真的', '太', '懒', '了', '。', '她', '打', '的', '很', '漂亮', '。'],
            self.test_processor.tokenize(
                sents[0], cws=True))

    def test_tokenize_and_untokenize(self):
        self.assertRaises(AssertionError, self.test_processor.tokenize, sents)
        self.assertRaises(
            AssertionError,
            self.test_processor.inverse_tokenize,
            sents[0]
        )

        for sent in sents:
            self.assertEqual(
                sent,
                self.test_processor.inverse_tokenize(
                    self.test_processor.tokenize(sent)
                )
            )

        self.assertEqual("我不理解这个问题！。？",
                         self.test_processor.inverse_tokenize(
                             self.test_processor.tokenize(
                                 "我不理解这个问题！。？"
                             ))
                         )

    def test_get_ner(self):
        sent = '小明想去吃螺蛳粉。'
        self.assertRaises(AssertionError, self.test_processor.get_ner, {})
        self.assertEqual(([], []), self.test_processor.get_ner(''))
        self.assertEqual(([('Nh', 0, 1)], ['Nh', 'Nh', 'O', 'O', 'O', 'O', 'O', 'O', 'O']),
                         self.test_processor.get_ner(sent))

    def test_get_pos_tag(self):
        sent = '小明想去吃螺蛳粉。'
        self.assertRaises(AssertionError, self.test_processor.get_pos_tag, {})
        self.assertEqual([], self.test_processor.get_pos_tag(''))
        self.assertEqual([['nh', 0, 1], ['v', 2, 2], ['v', 3, 3], ['v', 4, 4], ['n', 5, 7], ['wp', 8, 8]],
                         self.test_processor.get_pos_tag(sent))

    def test_get_dp(self):
        sent = '小明想去吃螺蛳粉。'
        self.assertRaises(AssertionError, self.test_processor.get_dp, {})
        self.assertEqual([], self.test_processor.get_dp(''))
        self.assertEqual([(1, 2, 'SBV'), (2, 0, 'HED'), (3, 4, 'ADV'), (4, 2, 'VOB'), (5, 4, 'VOB'), (6, 2, 'WP')],
                         self.test_processor.get_dp(sent))

    def test_sentence_tokenize(self):
        self.assertRaises(
            AssertionError,
            self.test_processor.sentence_tokenize,
            sents
        )
        self.assertEqual(
            2,
            len(self.test_processor.sentence_tokenize(sents[0]))
        )
        self.assertEqual(["我们正在玩乒乓球，你真的太懒了。", "她打的很漂亮。"],
                         self.test_processor.sentence_tokenize(sents[0]))

    # def test_get_synonyms(self):
    #     sent = '小明想去吃螺蛳粉。'
    #     self.assertRaises(AssertionError, self.test_processor.get_synonyms, [])
    #     self.assertEqual([[], ['务期', '推论', '揣测', '希', '企', '盘算', '幸', '企望', '期待', '叨念', '揆度', '祈',
    #                            '逻辑思维', '揆', '盼', '惦念', '揣度', '指望', '盼望', '可望', '寻味', '动脑筋', '瞩望',
    #                            '思念', '想望', '合计', '巴望', '感怀', '祈望', '怀恋', '冀', '思谋', '顾念', '仰望', '揣摩',
    #                            '虑', '以己度人', '惦记', '思想', '由此可知', '推度', '思慕', '想念', '思维', '眷恋', '欲',
    #                            '默想', '但愿', '祷', '怀想', '推理', '思忖', '想想', '想见', '琢磨', '梦想', '朝思暮想',
    #                            '期', '感念', '触景伤情', '企盼', '想来', '推求', '思', '望', '相思', '酌量', '要', '测度', '揣摸',
    #                            '希望', '考虑', '怀念', '构思', '寻思', '推测', '期望', '忖度', '眷念', '思考', '推想', '思辨', '心想',
    #                            '度', '审度', '思量', '沉凝', '巴', '愿意', '思虑', '思索', '纪念', '沉思', '推断', '冀望', '忖量',
    #                            '只求', '意在', '测算'], ['删', '夺', '离开', '走人', '往', '距', '删去', '刨除', '剔', '装', '串演',
    #                             '徊', '饰', '失去', '前往', '过去', '奔', '相差', '前去', '剔除', '除去', '芟除', '失掉', '距离', '开走',
    #                             '删除', '扮', '装扮', '失却', '失', '通往', '删减', '撤离', '撤出', '勾', '扮演', '抹', '错过', '背离',
    #                             '离', '转赴', '饰演', '造', '踅', '之', '赴', '偏离', '串', '错开', '相距', '走', '扮作', '去除', '离去'],
    #                             ['面临', '罹', '中', '剿灭', '横扫千军', '凭着', '蒙', '死仗', '歼敌', '啖', '歼灭', '歼', '动', '民以食为天',
    #                              '遭遇', '着', '深受', '蒙受', '受到', '未遭', '全歼', '遭到', '零吃', '遭逢', '备受', '丁', '遭受', '吃掉',
    #                              '解决', '遇', '屡遭', '自恃', '倍受', '消灭', '淘', '为', '损耗', '食', '负', '受', '茹', '遭劫', '挨',
    #                              '偏', '取给', '藉', '饱受', '让', '消耗', '惨遭', '攻歼', '耗费', '耗', '吃请', '于', '餐', '凭坚', '歼击',
    #                              '遭', '叫', '服', '饱尝', '被', '给', '磨耗', '用'], [], []],
    #                      self.test_processor.get_synonyms(sent))
    #
    # def test_get_antonyms(self):
    #     sent = '小明想去吃螺蛳粉。'
    #     self.assertRaises(AssertionError, self.test_processor.get_antonyms, [])
    #     self.assertEqual([[], [], ['归', '留', '来', '回', '还'], [], [], []],
    #                      self.test_processor.get_antonyms(sent))

if __name__ == "__main__":
    unittest.main()
