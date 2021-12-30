import unittest
from textflint.input.dataset import Dataset
from textflint.generation.generator.mrccn_generator import MRCCNGenerator
from textflint.input.component.sample import MRCCnSample

context = '众所周知，范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。范廷颂于1919年6月15日在越南宁平省天主教发艳教区出生；童年时接受良好教育后，被一位越南神父带到河内继续其学业。范廷颂于1940年在河内大修道院完成神学学业。' \
          '范廷颂于1949年6月6日在河内的主教座堂晋铎；及后被派到圣女小德兰孤儿院服务。1950年代，范廷颂在河内堂区创建移民接待中心以收容到河内避战的难民。1954年，法越战争结束，越南民主共和国建都河内，当时很多天主教神职人员逃至越南的南方，但范廷颂仍然留在河内。翌年管理圣若望小修院；惟在1960年因捍卫修院的自由、自治及拒绝政府在修院设政治课的要求而被捕。' \
          '1963年4月5日，教宗任命范廷颂为天主教北宁教区主教，同年8月15日就任；其牧铭为「我信天主的爱」。由于范廷颂被越南政府软禁差不多30年，因此他无法到所属堂区进行牧灵工作而专注研读等工作。范廷颂除了面对战争、贫困、被当局迫害天主教会等问题外，也秘密恢复修院、创建女修会团体等。1990年，教宗若望保禄二世在同年6月18日擢升范廷颂为天主教河内总教区宗座署理以填补该教区总主教的空缺。' \
          '1994年3月23日，范廷颂被教宗若望保禄二世擢升为天主教河内总教区总主教并兼天主教谅山教区宗座署理；同年11月26日，若望保禄二世擢升范廷颂为枢机。范廷颂在1995年至2001年期间出任天主教越南主教团主席。2003年4月26日，教宗若望保禄二世任命天主教谅山教区兼天主教高平教区吴光杰主教为天主教河内总教区署理主教；及至2005年2月19日，范廷颂因获批辞去总主教职务而荣休；吴光杰同日真除天主教河内总教区总主教职务。' \
          '范廷颂于2009年2月22日清晨在河内离世，享年89岁；其葬礼于同月26日上午在天主教河内总教区总主教座堂举行。钱能解决一切问题。 50.! @ # $ % ^ & * ( )'
data_sample = MRCCnSample(
    {'context': context, 'question': '范廷颂是什么时候被任为主教的?',
     'answers': [{"text": "1963年", "answer_start": 35},
                 {"text": "1963年", "answer_start": 35},
                 {"text": "1963年", "answer_start": 35}],
     'title': "范廷颂", 'is_impossible': False})
sample2 = MRCCnSample(
    {'context': " ", 'question': '范廷颂是什么时候被任为主教的?',
     'answers': [], 'title': "范廷颂", 'is_impossible': True})
sample3 = MRCCnSample(
    {'context': "! @ # $ % ^ & * ( )",
     'question': '范廷颂是什么时候被任为主教的?',
     'answers': [], 'title': "范廷颂", 'is_impossible': True})

dataset = Dataset('MRCCN')
dataset.load(data_sample)
dataset.extend([sample2, sample3])


class TestMRCGenerator(unittest.TestCase):

    def test_generate(self):
        # test task transformation
        trans_methods = [ "ModifyPos"]
        gene = MRCCNGenerator(trans_methods=trans_methods,
                            sub_methods=[])
        for original_samples, trans_rst, trans_type in gene.generate(dataset):
            self.assertEqual(1, len(trans_rst))
            for index in range(len(original_samples)):
                ori_sample = original_samples[index]
                tran_sample = trans_rst[index]
                self.assertEqual(ori_sample.get_words('question'),
                                 tran_sample.get_words('question'))
                ori_answers = ori_sample.get_answers()
                tran_answers = tran_sample.dump()['answers']
                for i in range(len(ori_answers)):
                    self.assertEqual(ori_answers[i]['text'],
                                     tran_answers[i]['text'])

        # test wrong trans_methods
        gene = MRCCNGenerator(trans_methods=["wrong_transform_method"],
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = MRCCNGenerator(trans_methods=["AddSubtree"],
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        gene = MRCCNGenerator(trans_methods="OOV",
                            sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))


if __name__ == "__main__":
    unittest.main()
