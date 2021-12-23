import unittest

from textflint.input.component.sample import MRCCnSample
from textflint.generation.transformation.MRC_cn import ModifyPos

context = '米尼科伊岛（Minicoy）位于印度拉克沙群岛中央直辖区最南端，是Lakshadweep县的一个城镇。它与拉克沙群岛隔九度海峡相望，与马尔代夫伊哈万迪富卢环礁隔八度海峡相望。总人口9495（2001年）。米尼科伊岛位于盐水胡东南部的一个有人居住的岛屿，全岛几乎被椰子树覆盖，唯一的地标是一座灯塔。Viringili位于米尼科伊岛西南侧，是一个长度不到两百米的小岛，曾被用作麻风病患者的驱逐地。该地2001年总人口9495人，其中男性4616人，女性4879人；0—6岁人口1129人，其中男571人，女558人；识字率81.95%，其中男性为83.51%，女性为80.47%。'
data_sample = MRCCnSample(
    {
        "context": "米尼科伊岛（Minicoy）位于印度拉克沙群岛中央直辖区最南端，是Lakshadweep县的一个城镇。它与拉克沙群岛隔九度海峡相望，与马尔代夫伊哈万迪富卢环礁隔八度海峡相望。总人口9495（2001年）。米尼科伊岛位于盐水胡东南部的一个有人居住的岛屿，全岛几乎被椰子树覆盖，唯一的地标是一座灯塔。Viringili位于米尼科伊岛西南侧，是一个长度不到两百米的小岛，曾被用作麻风病患者的驱逐地。该地2001年总人口9495人，其中男性4616人，女性4879人；0—6岁人口1129人，其中男571人，女558人；识字率81.95%，其中男性为83.51%，女性为80.47%。",
        "question": "米尼科伊岛（Minicoy）位于什么地方？",
        "answers": [
            {
                "text": "印度拉克沙群岛中央直辖区最南端",
                "answer_start": 16
            },
            {
                "text": "印度拉克沙群岛中央直辖区最南端",
                "answer_start": 16
            },
            {
                "text": "印度拉克沙群岛中央直辖区最南端",
                "answer_start": 16
            }
        ],
        "title": "米尼科伊岛",
        "is_impossible": False
    })
transformation = ModifyPos()


class TestModifyPos(unittest.TestCase):

    def test_ModifyPos(self):
        change = transformation.transform(data_sample, n=1)
        self.assertEqual(len(change), 1)
        original_sents = data_sample.get_sentences('context')
        trans_sents = change[0].get_sentences('context')
        changed = False
        for i in range(len(original_sents)):
            if trans_sents[i] != original_sents[i]:
                changed = True
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
