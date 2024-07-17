# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import cuda
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from transformers import BartTokenizer

sys.path.append("")
from dataset import SCIterator
from textcnn import TextCNN

filter_sizes = [1, 2, 3, 4, 5]
num_filters = [128, 128, 128, 128, 128]
device = 'cuda' if cuda.is_available() else 'cpu'
special_tokens = [{'bos_token': '<bos>'},
                  {'eos_token': '<eos>'}, {'sep_token': '<sep>'},
                  {'pad_token': '<pad>'}, {'unk_token': '<unk>'}]
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# for x in special_tokens:
#     tokenizer.add_special_tokens(x)


def main():
    parser = argparse.ArgumentParser('Evaluating Style Strength')
    parser.add_argument('-order', default=0, type=str, help='order')
    parser.add_argument('-style', default=0, type=int, help='from 0 to 1')
    parser.add_argument('-path_pred_A', default='path_pred_A', type=str, help='the path of style A predictions')
    parser.add_argument('-path_pred_B', default='path_pred_B', type=str, help='the path of style B predictions')
    parser.add_argument('-format', default=1, type=int, help='the format of predictions')
    parser.add_argument('-lowercase', action='store_true', dest="lowercase", default=False, help='whether to lowercase data')
    parser.add_argument('-onlyAB', action='store_true', dest="onlyAB", default=False, help='only A->B')
    parser.add_argument('-max_len', default=64, type=int, help='max tokens in a batch')
    parser.add_argument('-embed_dim', default=300, type=int, help='the embedding size')
    parser.add_argument('-dataset', default='em', type=str, help='the name of dataset')
    parser.add_argument('-model', default='textcnn', type=str, help='the name of model')
    parser.add_argument('-pretrained_path', default='pretrained_path', type=str, help='the path of pretrained model')
    parser.add_argument('-seed', default=42, type=int, help='pseudo random number seed')
    parser.add_argument('-batch_size', default=32, type=int, help='max sents in a batch')
    parser.add_argument("-dropout", default=0.5, type=float, help="Keep prob in dropout")

    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    test_src, test_tgt = [], []
    with open(opt.path_pred_A,'r') as f:
        for line in f.readlines() if opt.format == 1 else pd.read_csv(opt.path_pred_A).fillna('')['A (generated)'].tolist():
            if opt.lowercase:
                line = line.lower()
            test_src.append(tokenizer.encode(line.strip())[:opt.max_len])
    with open(opt.path_pred_B,'r') as f:
        for line in f.readlines() if opt.format == 1 else pd.read_csv(opt.path_pred_B).fillna('')['B (generated)'].tolist():
            if opt.lowercase:
                line = line.lower()
            test_tgt.append(tokenizer.encode(line.strip())[:opt.max_len])
    print('[Info] {} instances from src test set'.format(len(test_src)))
    print('[Info] {} instances from tgt test set'.format(len(test_tgt)))
    if opt.onlyAB:
        test_src = []
    test_loader = SCIterator(test_src, test_tgt, opt, tokenizer.pad_token_id)

    loss_fn = nn.CrossEntropyLoss()
    model = TextCNN(opt.embed_dim, len(tokenizer), filter_sizes,
                    num_filters, None, dropout=opt.dropout)
    model.to(device).eval()
    model.load_state_dict(torch.load(opt.pretrained_path))

    total_num = 0.
    total_acc = 0.
    total_loss = 0.
    with torch.no_grad():
        for i,batch in enumerate(test_loader):
            x_batch, y_batch = map(lambda x: x.to(device), batch)
            logits = model(x_batch)
            total_loss += loss_fn(logits, y_batch)
            _, y_hat = torch.max(logits,dim=-1)
            same = [float(p == q) for p, q in zip(y_batch, y_hat)]
            total_acc += sum(same)
            total_num += len(y_batch)

    print('Test: {}'.format('acc {:.4f}% | loss {:.4f}').format(
        total_acc / total_num * 100, total_loss / total_num))


if __name__ == '__main__':
    main()
