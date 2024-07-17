from data.datasets import ParallelRefDataset

import numpy as np
import pandas as pd
import random
import argparse

import torch
from torch.utils.data import DataLoader
from utils.utils import eval_ext

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser()

parser.add_argument('--method', type=str, dest="method", help='Method name.')
parser.add_argument('--format', type=int, dest="format", help='Output format.')
parser.add_argument('--lowercase_out', action='store_true', dest="lowercase_out", default=False, help='Whether to lowercase outputs.')
parser.add_argument('--lowercase_ref', action='store_true', dest="lowercase_ref", default=False, help='Whether to lowercase references.')
parser.add_argument('--style_a', type=str, dest="style_a", help='style A for the style transfer task (source style for G_ab).')
parser.add_argument('--style_b', type=str, dest="style_b", help='style B for the style transfer task (target style for G_ab).')
parser.add_argument('--lang', type=str, dest="lang", default='en', help='Dataset language.')
parser.add_argument('--pred_base_path', type=str, dest="pred_base_path", help='Predictions base path.')
parser.add_argument('--path_pred_A', type=str, dest="path_pred_A", help='Path to predictions in style A.')
parser.add_argument('--path_pred_B', type=str, dest="path_pred_B", help='Path to predictions in style B.')
parser.add_argument('--path_paral_A_test', type=str, dest="path_paral_A_test", help='Path to parallel dataset (style A) for evaluation.')
parser.add_argument('--path_paral_B_test', type=str, dest="path_paral_B_test", help='Path to parallel dataset (style B) for evaluation.')
parser.add_argument('--path_paral_test_ref', type=str, dest="path_paral_test_ref", help='Path to human references for evaluation.')
parser.add_argument('--n_references',  type=int, dest="n_references",  default=None, help='Number of human references for evaluation.')
parser.add_argument('--num_workers',type=int,  dest="num_workers", default=4,     help='Number of workers used for dataloaders.')
parser.add_argument('--pin_memory', action='store_true', dest="pin_memory",  default=False, help='Whether to pin memory for data on GPU during data loading.')
parser.add_argument('--max_sequence_length', type=int,  dest="max_sequence_length", default=64, help='Max sequence length')
parser.add_argument('--batch_size', type=int,  dest="batch_size",  default=64,     help='Batch size used during training.')
parser.add_argument('--pretrained_classifier_eval', type=str, dest="pretrained_classifier_eval", help='The folder to use as base path to load the pretrained classifier for metrics evaluation.')
parser.add_argument('--metrics', type=str, dest="metrics", help='Metrics to compute.')
parser.add_argument('--only_ab', action='store_true', dest="only_ab", default=False, help='Whether only A->B direction.')

args = parser.parse_args()
style_a = args.style_a
style_b = args.style_b

hyper_params = {}
print ("Arguments summary: \n ")
for key,value in vars(args).items():
    hyper_params[key] = value
    print (f"\t{key}:\t\t{value}")

def read_outputs(path_pred_A, path_pred_B, format):
    if format == 1:
        with open(path_pred_A, 'r') as f:
            pred_A = f.read().split('\n')
        with open(path_pred_B, 'r') as f:
            pred_B = f.read().split('\n')
        if pred_A[-1] == pred_B[-1] == '':
            pred_A, pred_B = pred_A[:-1], pred_B[:-1]
    elif format == 2:
        pred_A = pd.read_csv(path_pred_A).fillna('')['A (generated)'].tolist()
        pred_B = pd.read_csv(path_pred_B).fillna('')['B (generated)'].tolist()
    else:
        raise Exception(f'Format {format} is not supported.')
    return pred_A, pred_B


parallel_ds_testAB = ParallelRefDataset(dataset_format='line_file',
                                        style_src=style_a,
                                        style_ref=style_b,
                                        dataset_path_src=args.path_paral_A_test,
                                        dataset_path_ref=args.path_paral_test_ref,
                                        n_ref=args.n_references,
                                        separator_src='\n',
                                        separator_ref='\n')
    
parallel_ds_testBA = ParallelRefDataset(dataset_format='line_file',
                                        style_src=style_b,
                                        style_ref=style_a,
                                        dataset_path_src=args.path_paral_B_test,
                                        dataset_path_ref=args.path_paral_test_ref,
                                        n_ref=args.n_references,
                                        separator_src='\n',
                                        separator_ref='\n')

print (f"Parallel AB test: {len(parallel_ds_testAB)}")
print (f"Parallel BA test: {len(parallel_ds_testBA)}")

parallel_dl_testAB = DataLoader(parallel_ds_testAB,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=args.pin_memory,
                                collate_fn=ParallelRefDataset.customCollate)

parallel_dl_testBA = DataLoader(parallel_ds_testBA,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=args.pin_memory,
                                collate_fn=ParallelRefDataset.customCollate)


pred_A, pred_B = read_outputs(args.path_pred_A, args.path_pred_B, args.format)
assert len(pred_B) == len(parallel_ds_testAB)
assert len(pred_A) == len(parallel_ds_testBA)

eval_ext(args.metrics, pred_A, pred_B, parallel_dl_testAB, parallel_dl_testBA, args)