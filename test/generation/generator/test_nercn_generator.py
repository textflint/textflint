import unittest
from textflint.input.dataset import Dataset
from textflint.generation.generator.nercn_generator import NERCNGenerator

sample1={"x": ["上", "海", "浦", "东", "开", "发", "与", "法", "制", "建", "设", "同", "步"], "y": ["B-GPE", "E-GPE", "B-GPE", "E-GPE", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
sample2={"x": ["新", "华", "社", "上", "海", "二", "月", "十", "日", "电", "（", "记", "者", "谢", "金", "虎", "、", "张", "持", "坚", "）"], \
        "y": ["B-ORG", "M-ORG", "E-ORG", "B-GPE", "E-GPE", "O", "O", "O", "O", "O", "O", "O", "O", "B-PER", "M-PER", "E-PER", "O", "B-PER", "M-PER", "E-PER", "O"]}
sample3={"x": ["上", "海", "浦", "东", "近", "年", "来", "颁", "布", "实", "行", "了", "涉", "及", "经", "济", "、", "贸", "易", "、", "建", "设", "、", "规", "划", "、", "科", "技", "、", "文", "教", "等", "领", "域", "的", "七", "十", "一", "件", "法", "规", "性", "文", "件", "，", "确", "保", "了", "浦", "东", "开", "发", "的", "有", "序", "进", "行", "。"],\
     "y": ["B-GPE", "E-GPE", "B-GPE", "E-GPE", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-GPE", "E-GPE", "O", "O", "O", "O", "O", "O", "O", "O"]}


single_data_sample = [sample1]
data_samples = [sample1, sample2, sample3]
dataset = Dataset('NERCN')
single_dataset = Dataset('NERCN')
dataset.load(data_samples)
single_dataset.load(single_data_sample)
gene = NERCNGenerator()


class TestSpecialEntityTyposSwap(unittest.TestCase):

    def test_generate(self):
        # test wrong trans_methods
        gene = NERCNGenerator(trans_methods=["wrong_transform_method"],
                              sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))
        self.assertRaises(ValueError, next, gene.generate(dataset))

        gene = NERCNGenerator(trans_methods="OOV", sub_methods=[])
        self.assertRaises(ValueError, next, gene.generate(dataset))


if __name__ == "__main__":
    unittest.main()
