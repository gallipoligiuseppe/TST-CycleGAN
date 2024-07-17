from comet_ml import Experiment

from data.datasets import MonostyleDataset, ParallelRefDataset
from cyclegan_tst.models.CycleGANModel import CycleGANModel
from cyclegan_tst.models.DiscriminatorModel import DiscriminatorModel
from cyclegan_tst.models.GeneratorModel import GeneratorModel
from eval import *
from utils.utils import *

import argparse
import logging
import os
import numpy as np, pandas as pd
import random

import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                    PARSING PARAMs       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

parser = argparse.ArgumentParser()

# basic parameters
parser.add_argument('--style_a', type=str, dest="style_a", help='style A for the style transfer task (source style for G_ab).')
parser.add_argument('--style_b', type=str, dest="style_b", help='style B for the style transfer task (target style for G_ab).')
parser.add_argument('--lang', type=str, dest="lang", default='en', help='Dataset language.')
parser.add_argument('--max_samples_test',  type=int, dest="max_samples_test",  default=None, help='Max number of examples to retain from the test set. None for all available examples.')

parser.add_argument('--path_mono_A_test', type=str, dest="path_mono_A_test", help='Path to non-parallel dataset (style A) for test.')
parser.add_argument('--path_mono_B_test', type=str, dest="path_mono_B_test", help='Path to non-parallel dataset (style B) for test.')
parser.add_argument('--path_paral_A_test', type=str, dest="path_paral_A_test", help='Path to parallel dataset (style A) for test.')
parser.add_argument('--path_paral_B_test', type=str, dest="path_paral_B_test", help='Path to parallel dataset (style B) for test.')
parser.add_argument('--path_paral_test_ref', type=str, dest="path_paral_test_ref", help='Path to human references for test.')
parser.add_argument('--n_references',  type=int, dest="n_references",  default=None, help='Number of human references for test.')
parser.add_argument('--lowercase_ref', action='store_true', dest="lowercase_ref", default=False, help='Whether to lowercase references.')
parser.add_argument('--bertscore', action='store_true', dest="bertscore", default=True, help='Whether to compute BERTScore metric.')

parser.add_argument('--max_sequence_length', type=int,  dest="max_sequence_length", default=64, help='Max sequence length')

# Training arguments
parser.add_argument('--batch_size', type=int,  dest="batch_size",  default=64,     help='Batch size used during training.')
parser.add_argument('--num_workers',type=int,  dest="num_workers", default=4,     help='Number of workers used for dataloaders.')
parser.add_argument('--pin_memory', action='store_true', dest="pin_memory",  default=False, help='Whether to pin memory for data on GPU during data loading.')

parser.add_argument('--use_cuda_if_available', action='store_true', dest="use_cuda_if_available", default=False, help='Whether to use GPU if available.')

parser.add_argument('--generator_model_tag', type=str, dest="generator_model_tag", help='The tag of the model for the generator (e.g., "facebook/bart-base").')
parser.add_argument('--discriminator_model_tag', type=str, dest="discriminator_model_tag", help='The tag of the model discriminator (e.g., "distilbert-base-cased").')
parser.add_argument('--pretrained_classifier_eval', type=str, dest="pretrained_classifier_eval", help='The folder to use as base path to load the pretrained classifier for metrics evaluation.')

# arguments for saving the model and running test
parser.add_argument('--save_base_folder', type=str, dest="save_base_folder", help='The folder to use as base path to store model checkpoints')
parser.add_argument('--from_pretrained', type=str, dest="from_pretrained", default=None, help='The folder to use as base path to load model checkpoints')
parser.add_argument('--test_id', type=str, dest="test_id", default=None, help='Test ID')

# arguments for comet
parser.add_argument('--comet_logging', action='store_true', dest="comet_logging",   default=False, help='Set flag to enable comet logging')
parser.add_argument('--comet_key',       type=str,  dest="comet_key",       default=None,  help='Comet API key to log some metrics')
parser.add_argument('--comet_workspace', type=str,  dest="comet_workspace", default=None,  help='Comet workspace name (usually username in Comet, used only if comet_key is not None')
parser.add_argument('--comet_project_name',  type=str,  dest="comet_project_name",  default=None,  help='Comet experiment name (used only if comet_key is not None')
parser.add_argument('--exp_group', type=str, dest="exp_group", default=None, help='To group experiments on Comet')

args = parser.parse_args()

style_a = args.style_a
style_b = args.style_b
max_samples_test = args.max_samples_test

hyper_params = {}
print ("Arguments summary: \n ")
for key,value in vars(args).items():
    hyper_params[key] = value
    print (f"\t{key}:\t\t{value}")

if args.n_references is not None:
    parallel_ds_testAB = ParallelRefDataset(dataset_format='line_file',
                                            style_src=style_a,
                                            style_ref=style_b,
                                            dataset_path_src=args.path_paral_A_test,
                                            dataset_path_ref=args.path_paral_test_ref,
                                            n_ref=args.n_references,
                                            separator_src='\n',
                                            separator_ref='\n',
                                            max_dataset_samples=args.max_samples_test)
    
    parallel_ds_testBA = ParallelRefDataset(dataset_format='line_file',
                                            style_src=style_b,
                                            style_ref=style_a,
                                            dataset_path_src=args.path_paral_B_test,
                                            dataset_path_ref=args.path_paral_test_ref,
                                            n_ref=args.n_references,
                                            separator_src='\n',
                                            separator_ref='\n',
                                            max_dataset_samples=args.max_samples_test)
else:
    mono_ds_a_test = MonostyleDataset(dataset_format="line_file",
                                      style=style_a,
                                      dataset_path=args.path_mono_A_test,
                                      separator='\n',
                                      max_dataset_samples=args.max_samples_test)

    mono_ds_b_test = MonostyleDataset(dataset_format="line_file",
                                      style=style_b,
                                      dataset_path=args.path_mono_B_test,
                                      separator='\n',
                                      max_dataset_samples=args.max_samples_test)

if args.n_references is not None:
    print (f"Parallel AB test: {len(parallel_ds_testAB)}")
    print (f"Parallel BA test: {len(parallel_ds_testBA)}")
else:
    print (f"Mono A test: {len(mono_ds_a_test)}")
    print (f"Mono B test: {len(mono_ds_b_test)}")
print()

if args.n_references is not None:
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
    del parallel_ds_testAB, parallel_ds_testBA
else:
    mono_dl_a_test = DataLoader(mono_ds_a_test,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=args.pin_memory)

    mono_dl_b_test = DataLoader(mono_ds_b_test,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=args.pin_memory)
    del mono_ds_a_test, mono_ds_b_test

if args.n_references is not None:
    print (f"Parallel AB test (batches): {len(parallel_dl_testAB)}")
    print (f"Parallel BA test (batches): {len(parallel_dl_testBA)}")
else:
    print (f"Mono A test (batches): {len(mono_dl_a_test)}")
    print (f"Mono B test (batches): {len(mono_dl_b_test)}")

if args.comet_logging :
    experiment = Experiment(api_key=args.comet_key,
                            project_name=args.comet_project_name,
                            workspace=args.comet_workspace)
    experiment.log_parameters(hyper_params)
else:
    experiment = None

if args.use_cuda_if_available:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")

if args.from_pretrained is not None:
    if 'epoch' in args.from_pretrained:
        checkpoints_paths = [args.from_pretrained]
    else:
        checkpoints_paths = sorted([dir for dir in os.listdir(args.from_pretrained) if dir.startswith('epoch')], key=lambda dir: int(dir.split('_')[1]))
        checkpoints_paths = [args.from_pretrained+path+'/' for path in checkpoints_paths]
    epochs = [int(path.split('_')[-1][:-1]) for path in checkpoints_paths]
else:
    checkpoints_paths, epochs = [''], [0]

''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                        TEST       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

for checkpoint, epoch in zip(checkpoints_paths, epochs):
    if args.from_pretrained is not None:
        G_ab = GeneratorModel(args.generator_model_tag, f'{checkpoint}G_ab/', max_seq_length=args.max_sequence_length)
        G_ba = GeneratorModel(args.generator_model_tag, f'{checkpoint}G_ba/', max_seq_length=args.max_sequence_length)
        print('Generator pretrained models loaded correctly')
        D_ab = DiscriminatorModel(args.discriminator_model_tag, f'{checkpoint}D_ab/', max_seq_length=args.max_sequence_length)
        D_ba = DiscriminatorModel(args.discriminator_model_tag, f'{checkpoint}D_ba/', max_seq_length=args.max_sequence_length)
        print('Discriminator pretrained models loaded correctly')
    else:
        G_ab = GeneratorModel(args.generator_model_tag, max_seq_length=args.max_sequence_length)
        G_ba = GeneratorModel(args.generator_model_tag, max_seq_length=args.max_sequence_length)
        print('Generator pretrained models not loaded - Initial weights will be used')
        D_ab = DiscriminatorModel(args.discriminator_model_tag, max_seq_length=args.max_sequence_length)
        D_ba = DiscriminatorModel(args.discriminator_model_tag, max_seq_length=args.max_sequence_length)
        print('Discriminator pretrained models not loaded - Initial weights will be used')
    
    cycleGAN = CycleGANModel(G_ab, G_ba, D_ab, D_ba, None, device=device)
    evaluator = Evaluator(cycleGAN, args, experiment)

    if args.n_references is not None:
        evaluator.run_eval_ref(epoch, epoch, 'test', parallel_dl_testAB, parallel_dl_testBA)
    else:
        evaluator.run_eval_mono(epoch, epoch, 'test', mono_dl_a_test, mono_dl_b_test)

print('End checkpoint(s) test...')
