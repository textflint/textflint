import os
import json
import csv
import random
import numpy as np
import torch


def data_loader(data_path):
    data_inputs = []
    # sa_data_set = Dataset(task='SA')

    with open(data_path, "r+") as fo:
        for line in fo.readlines():
            data_sample = json.loads(line)
            data_inputs.append({
                'x': data_sample['reviewText'],
                'y': 'positive' if data_sample['overall'] > 4 else 'negative'
            })
    # sa_data_set.load(data_inputs)

    return data_inputs


def data_loader_csv(data_path):
    data_inputs = []

    with open(data_path, "r+") as fo:
        all_lines = csv.reader(fo)
        for line in list(all_lines)[1:]:
            data_inputs.append({
                'x': line[-1],
                'y': 'positive' if int(line[1]) == 1 else 'negative'
            })

    random.shuffle(data_inputs)
    return data_inputs


def map_sample(sa_sample, tokenizer, label2id):
    x = sa_sample['x']
    # tokenize and convert token to id
    token_ids = tokenizer.encode(x)
    label_id = label2id[sa_sample['y']]

    return token_ids, label_id


def train_iter(data_set, batch_size, tokenizer, label2id):
    i = 0

    while i < len(data_set):
        batch = data_set[i: i + batch_size]
        i += batch_size
        input_ids = []
        label_ids = []

        for x in batch:
            token_ids, label_id = map_sample(x, tokenizer, label2id)
            input_ids.append(token_ids)
            label_ids.append(label_id)

        yield np.array(input_ids), np.array(label_ids)


def torch_save(model, save_dir, save_prefix, steps=0):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pkl'.format(save_prefix, steps)
    torch.save(model.state_dict(), open(save_path, "wb"))


def tf_save(model, save_dir, save_prefix, steps=0):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}_wx'.format(save_prefix, steps)
    # TODO
    model.save_weights(save_path)
