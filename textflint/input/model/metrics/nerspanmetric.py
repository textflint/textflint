from abc import ABC


class NERSpanMetric(ABC):
    def __init__(self, encoding_type=None, tag_vocab=None):
        self.all_word = 0.0
        self.pred_word = 0.0
        self.pred_real_word = 0.0
        self.encoding_type = encoding_type
        if self.encoding_type == 'bmes':
            self.tag_to_span_func = self._bmes_tag_to_spans
        elif self.encoding_type == 'bio':
            self.tag_to_span_func = self._bio_tag_to_spans
        elif self.encoding_type == 'bioes':
            self.tag_to_span_func = self._bioes_tag_to_spans
        else:
            raise ValueError(
                "Only support 'bio', 'bmes', 'bmeso', 'bioes' type."
            )
        self.tag_vocab = tag_vocab

    def get_metric(self):
        p = self.pred_real_word / self.pred_word
        r = self.pred_real_word / self.all_word
        return {'f': 2 * (p * r) / (p + r), 'pre': p, 'rec': r}

    def evaluate(self, pred, target, seq_len):
        # for i in range(int(seq_len.shape[0])):
        for i in range(len(seq_len)):
            if self.tag_vocab:
                Pred = ['O'] * int(seq_len[i])
                Target = ['O'] * int(seq_len[i])
                for j, k in enumerate(pred[i]):
                    if k >= 0 and k < seq_len[i]:
                        Pred[j] = self.tag_vocab[int(k)]
                for j, k in enumerate(target[i]):
                    if k >= 0 and k < seq_len[i]:
                        Target[j] = self.tag_vocab[int(k)]
                Pred = self.tag_to_span_func(Pred)
                Target = self.tag_to_span_func(Target)
            else:
                Pred = self.tag_to_span_func(pred[i])
                Target = self.tag_to_span_func(target[i])

            self.all_word += len(Target)
            self.pred_word += len(Pred)
            for (i, j) in Pred:
                for (n, m) in Target:
                    if i == n and j == m:
                        self.pred_real_word += 1
                        break

    @staticmethod
    def _bmes_tag_to_spans(tags, ignore_labels=None):
        r"""
        给定一个tags的lis，比如['S-song', 'B-singer', 'M-singer', 'E-singer', 'S-moive', 'S-actor']。
        返回[('song', (0, 1)), ('singer', (1, 4)), ('moive', (4, 5)), ('actor', (5, 6))] (左闭右开区间)
        也可以是单纯的['S', 'B', 'M', 'E', 'B', 'M', 'M',...]序列

        :param tags: List[str],
        :param ignore_labels: List[str], 在该list中的label将被忽略
        :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
        """
        ignore_labels = set(ignore_labels) if ignore_labels else set()

        spans = []
        prev_bmes_tag = None
        for idx, tag in enumerate(tags):
            tag = tag.lower()
            bmes_tag, label = tag[:1], tag[2:]
            if bmes_tag in ('b', 's'):
                spans.append((label, [idx, idx]))
            elif bmes_tag in ('m', 'e') and prev_bmes_tag in (
            'b', 'm') and label == spans[-1][0]:
                spans[-1][1][1] = idx
            else:
                spans.append((label, [idx, idx]))
            prev_bmes_tag = bmes_tag
        return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans
                if span[0] not in ignore_labels]

    @staticmethod
    def _bioes_tag_to_spans(tags, ignore_labels=None):
        r"""
        给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'E-singer', 'O', 'O']。
        返回[('singer', (1, 4))] (左闭右开区间)

        :param tags: List[str],
        :param ignore_labels: List[str], 在该list中的label将被忽略
        :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
        """
        ignore_labels = set(ignore_labels) if ignore_labels else set()

        spans = []
        prev_bioes_tag = None
        for idx, tag in enumerate(tags):
            tag = tag.lower()
            bioes_tag, label = tag[:1], tag[2:]
            if bioes_tag in ('b', 's'):
                spans.append((label, [idx, idx]))
            elif bioes_tag in ('i', 'e') and prev_bioes_tag in ('b', 'i') and label == spans[-1][0]:
                spans[-1][1][1] = idx
            elif bioes_tag == 'o':
                pass
            else:
                spans.append((label, [idx, idx]))
            prev_bioes_tag = bioes_tag
        return [(span[0], (span[1][0], span[1][1] + 1))
                for span in spans
                if span[0] not in ignore_labels
                ]

    @staticmethod
    def _bio_tag_to_spans(tags, ignore_labels=None):
        r"""
        给定一个tags的lis，比如['O', 'B-singer', 'I-singer', 'I-singer', 'O', 'O']。
            返回[('singer', (1, 4))] (左闭右开区间)

        :param tags: List[str],
        :param ignore_labels: List[str], 在该list中的label将被忽略
        :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
        """
        ignore_labels = set(ignore_labels) if ignore_labels else set()

        spans = []
        prev_bio_tag = None
        for idx, tag in enumerate(tags):
            tag = tag.lower()
            bio_tag, label = tag[:1], tag[2:]
            if bio_tag == 'b':
                spans.append((label, [idx, idx]))
            elif bio_tag == 'i' and prev_bio_tag in ('b', 'i') and label == spans[-1][0]:
                spans[-1][1][1] = idx
            elif bio_tag == 'o':  # o tag does not count
                pass
            else:
                spans.append((label, [idx, idx]))
            prev_bio_tag = bio_tag
        return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans if span[0] not in ignore_labels]