import os
import numpy as np
import torch
from torch.utils.data import Dataset


def read_dict(dataset):
    entity2id = dict()
    relation2id = dict()
    entity_path = os.path.join("data", dataset, "entities.dict")
    relation_path = os.path.join("data", dataset, "relations.dict")
    with open(entity_path) as fin:
        for line in fin:
            eid, entity = line.strip().split()
            entity2id[entity] = int(eid)
    with open(relation_path) as fin:
        for line in fin:
            rid, relation = line.strip().split()
            relation2id[relation] = int(rid)
    return entity2id, relation2id


def read_triples(dataset, entity2id, relation2id, mode):
    triples = []
    triples_path = os.path.join("data", dataset, mode)
    with open(triples_path) as fin:
        for line in fin:
            h, r, t = line.strip().split()
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


class TrainDataset(Dataset):
    def __init__(self, triples, entity_num, relation_num, negative_sample_size, mode):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        positive_sample = self.triples[index]
        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = np.sqrt(1 / subsampling_weight)

        negative_sample = self.negative_sample(positive_sample, self.mode)

        positive_sample = torch.tensor(positive_sample)
        negative_sample = torch.tensor(negative_sample)
        subsampling_weight = torch.tensor(subsampling_weight)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    def negative_sample(self, positive_sample, mode):
        head, relation, tail = positive_sample

        negative_sample_list = []
        negative_sample_size = 0

        while negative_sample_size < self.negative_sample_size:
            negative_sample = np.random.randint(self.entity_num, size=self.negative_sample_size)

            if mode == 'head':
                mask = np.in1d(negative_sample, self.true_head[(relation, tail)], invert=True)
            elif mode == 'tail':
                mask = np.in1d(negative_sample, self.true_tail[(head, relation)], invert=True)
            else:
                raise ValueError('Negative sample mode %s not supported' % mode)

            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        return negative_sample

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, entity_num, relation_num, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        positive_sample = self.triples[index]

        entity_filter = self.entity_filter(positive_sample, self.mode)
        filter_bias, negative_sample = zip(*entity_filter)

        positive_sample = torch.tensor(positive_sample)
        negative_sample = torch.tensor(negative_sample)
        filter_bias = torch.tensor(filter_bias)

        return positive_sample, negative_sample, filter_bias, self.mode

    def entity_filter(self, positive_sample, mode):
        head, relation, tail = positive_sample

        if mode == 'head':
            entity_filter = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                             else (-1, head) for rand_head in range(self.entity_num)]
            entity_filter[head] = (0, head)
        elif mode == 'tail':
            entity_filter = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                             else (-1, tail) for rand_tail in range(self.entity_num)]
            entity_filter[tail] = (0, tail)
        else:
            raise ValueError('Negative sample mode %s not supported' % mode)

        return entity_filter

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode
