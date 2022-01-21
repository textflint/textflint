import unittest

from textflint.input.component.sample import MRCCnSample
from textflint.generation.transformation.MRCCN import PerturbQuestion
import random
seed = 123
random.seed(seed)

context = '罗宾·雨果·吉布（Robin Hugh Gibb，），英国歌手、词曲作者及唱片监制，出生于英国皇家属地曼岛的首府道格拉斯，双亲都是英国人。他在吉布斯家族五姐弟排行第三（在1969年，因为罗宾跟巴里和经理人吵架而单飞，姐姐 Lesley 成为代唱），是双胞胎弟弟莫里斯（Maurice Ernest Gibb）的哥哥。他与莫里斯和哥哥巴里·吉布（Barry Alan Crompton Gibb）在澳洲组成的乐队Bee Gees闻名世界，是20世纪摇滚乐和迪斯科高峰时间的代表乐团，开创了多种新颖的演唱方式，影响十分深远，成为史上最成功的其中一支乐队。在与大肠癌及肝癌长期搏斗后，罗宾于2012年5月20日逝世，享年62岁。'
data_sample = MRCCnSample(
    {
        "context": "罗宾·雨果·吉布（Robin Hugh Gibb，），英国歌手、词曲作者及唱片监制，出生于英国皇家属地曼岛的首府道格拉斯，双亲都是英国人。他在吉布斯家族五姐弟排行第三（在1969年，因为罗宾跟巴里和经理人吵架而单飞，姐姐 Lesley 成为代唱），是双胞胎弟弟莫里斯（Maurice Ernest Gibb）的哥哥。他与莫里斯和哥哥巴里·吉布（Barry Alan Crompton Gibb）在澳洲组成的乐队Bee Gees闻名世界，是20世纪摇滚乐和迪斯科高峰时间的代表乐团，开创了多种新颖的演唱方式，影响十分深远，成为史上最成功的其中一支乐队。在与大肠癌及肝癌长期搏斗后，罗宾于2012年5月20日逝世，享年62岁。",
        "question": "罗宾在吉布斯家族五姐弟排行第几？",
        "answers": [
            {
                "text": "第三",
                "answer_start": 81
            },
            {
                "text": "第三",
                "answer_start": 81
            },
            {
                "text": "第三",
                "answer_start": 81
            },
        ],
        "title": "罗宾·吉布",
        "is_impossible": False
    })
transformation = PerturbQuestion(transform_method='mlm_suggestion')

class TestPerturbQuestion(unittest.TestCase):
    @unittest.skip("Manual test")

    def test_PerturbQuestion(self):

        change = transformation.transform(data_sample, n=1)
        trans_sents = change[0].get_text('question')

        self.assertTrue(trans_sents != data_sample.get_text('question'))



if __name__ == "__main__":
    unittest.main()
