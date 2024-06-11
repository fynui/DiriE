import logging

import numpy as np

import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.distributions import Dirichlet, kl_divergence

from torch.utils.data import DataLoader

from dataloader import TestDataset


class DiriE(Module):

    def __init__(self, entity_num, relation_num, hidden_dim, negative_sample_size=2, gamma=0):
        super(DiriE, self).__init__()
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.hidden_dim = hidden_dim
        self.entity_dim = hidden_dim * 2
        self.relation_dim = hidden_dim * 2
        self.negative_sample_size = negative_sample_size
        self.gamma = gamma
        self.epsilon = 2.0

        embedding_range = (self.gamma + self.epsilon) / self.hidden_dim

        self.entity_embedding = Parameter(torch.zeros(entity_num, self.entity_dim))
        self.relation_embedding = Parameter(torch.zeros(relation_num, self.relation_dim))
        torch.nn.init.uniform_(self.entity_embedding, -embedding_range, embedding_range)
        torch.nn.init.uniform_(self.relation_embedding, -embedding_range, embedding_range)

    def model(self, sample, mode='single'):

        if mode == 'single':
            h, r, t = sample[:, 0], sample[:, 1], sample[:, 2]

            head = torch.index_select(self.entity_embedding, dim=0, index=h)
            relation = torch.index_select(self.relation_embedding, dim=0, index=r)
            tail = torch.index_select(self.entity_embedding, dim=0, index=t)

        elif mode == 'head':
            positive_sample, negative_sample = sample
            h, r, t = positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]
            batch_size = positive_sample.size(0)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=negative_sample.reshape(-1)
            ).reshape(batch_size, -1, self.entity_dim)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=r
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=t
            ).unsqueeze(1)

        elif mode == 'tail':
            positive_sample, negative_sample = sample
            h, r, t = positive_sample[:, 0], positive_sample[:, 1], positive_sample[:, 2]
            batch_size = positive_sample.size(0)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=h
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding, dim=0, index=r
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=negative_sample.reshape(-1)
            ).reshape(batch_size, -1, self.entity_dim)

        elif mode == 'relation':
            h, r, t = sample[:, 0], sample[:, 1], sample[:, 2]
            batch_size = sample.size(0)

            head = torch.index_select(
                self.entity_embedding, dim=0, index=h
            ).unsqueeze(1)

            all_relation = self.relation_embedding.unsqueeze(0)

            tail = torch.index_select(
                self.entity_embedding, dim=0, index=t
            ).unsqueeze(1)

            # adversarial relation subsample
            score = self.score(head, all_relation, tail)
            seq = torch.sort(score, dim=1, descending=True)
            values, indices = seq[0], seq[1]
            mask = ~torch.eq(indices, r.unsqueeze(dim=1))

            values = torch.masked_select(
                values, mask
            ).reshape(batch_size, self.relation_num - 1)[:, :self.negative_sample_size]
            props = torch.softmax(values, dim=1)

            return values, props

        else:
            raise ValueError('mode %s not supported' % mode)

        return self.score(head, relation, tail)

    def score(self, head, relation, tail):
        head = F.softplus(head)
        relation = F.softplus(relation)
        tail = F.softplus(tail)

        dist = self.dist(head, relation, tail)
        return self.gamma - dist

    @staticmethod
    def dist(head: torch.Tensor, relation: torch.Tensor, tail: torch.Tensor):
        head_p, head_q = head.chunk(2, dim=-1)
        tail_p, tail_q = tail.chunk(2, dim=-1)
        relation_f, relation_b = relation.chunk(2, dim=-1)

        conjugate_f = head_p + relation_f
        conjugate_b = tail_p + relation_b

        dist1 = kl_divergence(Dirichlet(tail_q), Dirichlet(conjugate_f))
        dist2 = kl_divergence(Dirichlet(head_q), Dirichlet(conjugate_b))
        return dist1 + dist2

    def train_step(self, optimizer, sample, args):

        self.train()

        positive_sample, negative_sample, subsampling_weight, mode = sample

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        positive_score = self.model(positive_sample)
        positive_loss = -F.logsigmoid(positive_score)

        negative_score = self.model((positive_sample, negative_sample), mode=mode)
        negative_loss = -F.logsigmoid(-negative_score).mean(dim=1)

        values, props = self.model(positive_sample, mode='relation')
        negative_relation_score = -F.logsigmoid(-values) * props
        relation_weight = 0.01
        negative_relation_loss = negative_relation_score.sum(dim=1) * relation_weight

        if args.uni_weight:
            positive_loss = positive_loss.mean()
            negative_loss = negative_loss.mean()
            negative_relation_loss = negative_relation_loss.mean()
        else:
            positive_loss = (subsampling_weight * positive_loss).sum() / subsampling_weight.sum()
            negative_loss = (subsampling_weight * negative_loss).sum() / subsampling_weight.sum()
            negative_relation_loss = (subsampling_weight * negative_relation_loss).sum() / subsampling_weight.sum()

        loss = positive_loss + negative_loss + negative_relation_loss
        # loss = positive_loss + negative_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log = {
            'positive_loss': positive_loss.item(),
            'negative_loss': negative_loss.item(),
            'loss': loss.item()
        }

        return log

    def test_entity(self, test_triples, all_true_triples, args):

        self.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                self.entity_num,
                self.relation_num,
                mode='head'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                self.entity_num,
                self.relation_num,
                mode='tail'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []
        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score = self.model((positive_sample, negative_sample), mode)
                    score += filter_bias * self.gamma

                    arg_sort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    ranking = torch.eq(positive_arg.unsqueeze(1), arg_sort).nonzero()
                    ranks = ranking[:, 1] + 1

                    for i in range(batch_size):
                        rank = ranks[i].item()
                        logs.append({
                            'MRR': 1.0 / rank,
                            'MR': float(rank),
                            'HITS@1': 1.0 if rank <= 1 else 0.0,
                            'HITS@3': 1.0 if rank <= 3 else 0.0,
                            'HITS@10': 1.0 if rank <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))
                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
