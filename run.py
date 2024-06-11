import argparse
import logging
import os

import torch

from torch.utils.data import DataLoader
from dataloader import TrainDataset, TestDataset
from dataloader import read_dict, read_triples

from model import DiriE


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--dataset', type=str, default=None)
    
    parser.add_argument('-n', '--negative_sample_size', default=8, type=int)
    parser.add_argument('-d', '--hidden_dim', default=64, type=int)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-g', '--gamma', default=20.0, type=float)
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float)
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-cpu', '--cpu_num', default=12, type=int)
    parser.add_argument('-save', '--save_path', default='output', type=str)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--valid_steps', default=50, type=int)

    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')
    
    return parser.parse_args(args)


def set_logger(args):
    log_file = os.path.join(args.save_path, '{}.log'.format(args.dataset))

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))

        
def main(args):
    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/valid/test mode must be choosed.')
    if args.dataset is None:
        raise ValueError('one of dataset must be choosed.')
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args)

    entity2id, relation2id = read_dict(args.dataset)
    entity_num = len(entity2id)
    relation_num = len(relation2id)

    logging.info('dataset: %s' % args.dataset)
    logging.info('#entity: %d' % entity_num)
    logging.info('#relation: %d' % relation_num)
    
    train_triples = read_triples(args.dataset, entity2id, relation2id, "train.txt")
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triples(args.dataset, entity2id, relation2id, "valid.txt")
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triples(args.dataset, entity2id, relation2id, "test.txt")
    logging.info('#test: %d' % len(test_triples))

    all_true_triples = train_triples + valid_triples + test_triples
    
    model = DiriE(
        entity_num=entity_num,
        relation_num=relation_num,
        hidden_dim=args.hidden_dim,
        negative_sample_size=args.negative_sample_size,
        gamma=args.gamma
    )
    
    logging.info('Model Parameter Configuration:')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        model = model.cuda()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )

    logging.info('Start Training...')
    logging.info('negative_sample_size = %d' % args.negative_sample_size)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('learning_rate = %f' % args.learning_rate)

    for epoch in range(args.max_epochs):
        if epoch % 2 == 0:
            mode = 'head'
        else:
            mode = 'tail'

        train_dataset = TrainDataset(train_triples, entity_num, relation_num, args.negative_sample_size, mode)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=max(1, args.cpu_num // 2), collate_fn=TrainDataset.collate_fn)

        training_logs = []
        for _, sample in enumerate(train_dataloader):
            log = model.train_step(optimizer, sample, args)
            training_logs.append(log)

        metrics = {}
        for metric in training_logs[0].keys():
            metrics[metric] = sum([log[metric] for log in training_logs]) / len(training_logs)
        log_metrics('Training average', epoch, metrics)

        if epoch % args.valid_steps == 0:
            logging.info('Evaluating on Valid Dataset...')
            metrics = model.test_entity(valid_triples, all_true_triples, args)
            log_metrics('Valid', epoch, metrics)
    
    if args.do_train:
        logging.info('Evaluating on Train Dataset...')
        metrics = model.test_entity(train_triples, all_true_triples, args)
        log_metrics('Train', args.max_epochs, metrics)

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = model.test_entity(valid_triples, all_true_triples, args)
        log_metrics('Valid', args.max_epochs, metrics)
    
    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = model.test_entity(test_triples, all_true_triples, args)
        log_metrics('Test', args.max_epochs, metrics)


if __name__ == '__main__':
    main(parse_args())
