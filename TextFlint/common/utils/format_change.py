import json


class format_changer(object):
    def __init__(self, src_file):
        self.src_file = src_file

    def change_format(self):
        if 'nyt' in self.src_file:
            ret = self.nyt10_engine()  # txt
        elif 'tacred' in self.src_file:
            ret = self.tacred_engine()  # json
        else:
            raise Exception('Unsupported dataset!')
        return ret

    def nyt10_engine(self):
        data = []
        with open(self.src_file) as infile:
            for line in infile:
                changed_line = dict()
                line = eval(line)
                changed_line['x'] = line['text']
                changed_line['y'] = line['relation']
                head_span = line['h']['pos']
                tail_span = line['t']['pos']
                changed_line['subj'] = (
                    line['text'][head_span[0]:head_span[1]])
                changed_line['obj'] = (line['text'][tail_span[0]:tail_span[1]])
                data.append(changed_line)
            return data

    def tacred_engine(self):
        data = []
        with open(self.src_file) as infile:
            lines = json.load(infile)
        for line in lines:
            changed_line = dict()
            changed_line['x'] = ' '.join(line['token'])
            changed_line['y'] = line['relation']
            head_span = (line['subj_start'], line['subj_end'] + 1)
            tail_span = (line['obj_start'], line['obj_end'] + 1)
            changed_line['subj'] = ' '.join(
                line['token'][head_span[0]:head_span[1]])
            changed_line['obj'] = ' '.join(
                line['token'][tail_span[0]:tail_span[1]])
            data.append(changed_line)
        return data
