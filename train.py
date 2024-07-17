from comet_ml import Experiment, ExistingExperiment

from data.datasets import MonostyleDataset, ParallelRefDataset
from cyclegan_tst.models.CycleGANModel import CycleGANModel
from cyclegan_tst.models.GeneratorModel import GeneratorModel
from cyclegan_tst.models.DiscriminatorModel import DiscriminatorModel
from cyclegan_tst.models.ClassifierModel import ClassifierModel
from eval import *
from utils.utils import *

import argparse
import logging
from tqdm import tqdm
import os, sys, time
import pickle
import numpy as np, pandas as pd
import random

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

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
parser.add_argument('--max_samples_train', type=int, dest="max_samples_train", default=None, help='Max number of examples to retain from the training set. None for all available examples.')
parser.add_argument('--max_samples_eval',  type=int, dest="max_samples_eval",  default=None, help='Max number of examples to retain from the evaluation set. None for all available examples.')
parser.add_argument('--nonparal_same_size', action='store_true', dest="nonparal_same_size",  default=False, help='Whether to reduce non-parallel data to same size.')

parser.add_argument('--path_mono_A', type=str, dest="path_mono_A", help='Path to monostyle dataset (style A) for training.')
parser.add_argument('--path_mono_B', type=str, dest="path_mono_B", help='Path to monostyle dataset (style B) for training.')

parser.add_argument('--path_mono_A_eval', type=str, dest="path_mono_A_eval", help='Path to non-parallel dataset (style A) for evaluation.')
parser.add_argument('--path_mono_B_eval', type=str, dest="path_mono_B_eval", help='Path to non-parallel dataset (style B) for evaluation.')
parser.add_argument('--path_paral_A_eval', type=str, dest="path_paral_A_eval", help='Path to parallel dataset (style A) for evaluation.')
parser.add_argument('--path_paral_B_eval', type=str, dest="path_paral_B_eval", help='Path to parallel dataset (style B) for evaluation.')
parser.add_argument('--path_paral_eval_ref', type=str, dest="path_paral_eval_ref", help='Path to human references for evaluation.')
parser.add_argument('--n_references',  type=int, dest="n_references",  default=None, help='Number of human references for evaluation.')
parser.add_argument('--lowercase_ref', action='store_true', dest="lowercase_ref", default=False, help='Whether to lowercase references.')
parser.add_argument('--bertscore', action='store_true', dest="bertscore", default=True, help='Whether to compute BERTScore metric.')

parser.add_argument('--max_sequence_length', type=int,  dest="max_sequence_length", default=64, help='Max sequence length')

# Training arguments
parser.add_argument('--batch_size', type=int,  dest="batch_size",  default=64,     help='Batch size used during training.')
parser.add_argument('--shuffle',    action='store_true', dest="shuffle",     default=False, help='Whether to shuffle the training/eval set or not.')
parser.add_argument('--num_workers',type=int,  dest="num_workers", default=4,     help='Number of workers used for dataloaders.')
parser.add_argument('--pin_memory', action='store_true', dest="pin_memory",  default=False, help='Whether to pin memory for data on GPU during data loading.')

parser.add_argument('--use_cuda_if_available', action='store_true', dest="use_cuda_if_available", default=False, help='Whether to use GPU if available.')

parser.add_argument('--learning_rate',     type=float, dest="learning_rate",     default=5e-5,     help='Initial learning rate (e.g., 5e-5).')
parser.add_argument('--epochs',            type=int,   dest="epochs",            default=10,       help='The number of training epochs.')
parser.add_argument('--lr_scheduler_type', type=str,   dest="lr_scheduler_type", default="linear", help='The scheduler used for the learning rate management.')
parser.add_argument('--warmup', action='store_true', dest="warmup", default=False, help='Whether to apply warmup.')
parser.add_argument('--lambdas', type=str,   dest="lambdas", default="1|1|1|1|1|1", help='Lambdas for loss-weighting.')

parser.add_argument('--generator_model_tag', type=str, dest="generator_model_tag", help='The tag of the model for the generator (e.g., "facebook/bart-base").')
parser.add_argument('--discriminator_model_tag', type=str, dest="discriminator_model_tag", help='The tag of the model discriminator (e.g., "distilbert-base-cased").')
parser.add_argument('--pretrained_classifier_model', type=str, dest="pretrained_classifier_model", help='The folder to use as base path to load the pretrained classifier for classifier-guided loss.')
parser.add_argument('--pretrained_classifier_eval', type=str, dest="pretrained_classifier_eval", help='The folder to use as base path to load the pretrained classifier for metrics evaluation.')

# arguments for saving the model and running evaluation
parser.add_argument('--save_base_folder', type=str, dest="save_base_folder", help='The folder to use as base path to store model checkpoints')
parser.add_argument('--from_pretrained', type=str, dest="from_pretrained", default=None, help='The folder to use as base path to load model checkpoints')
parser.add_argument('--save_steps',       type=int, dest="save_steps",       help='How many training epochs between two checkpoints.')
parser.add_argument('--eval_strategy',    type=str, dest="eval_strategy",    help='Evaluation strategy for the model (either epochs or steps)')
parser.add_argument('--eval_steps',       type=int, dest="eval_steps",       help='How many training steps between two evaluations.')
parser.add_argument('--additional_eval',       type=int, dest="additional_eval", default=0, help='Whether to perform evaluation at the half of the first N epochs.')

# temporary arguments to control execution
parser.add_argument('--control_file', type=str, dest="control_file", default=None, help='The path of the file to control execution (e.g., whether to stop)')
parser.add_argument('--lambda_file', type=str, dest="lambda_file", default=None, help='The path of the file to define lambdas')

# arguments for comet
parser.add_argument('--comet_logging', action='store_true', dest="comet_logging",   default=False, help='Set flag to enable comet logging')
parser.add_argument('--comet_key',       type=str,  dest="comet_key",       default=None,  help='Comet API key to log some metrics')
parser.add_argument('--comet_workspace', type=str,  dest="comet_workspace", default=None,  help='Comet workspace name (usually username in Comet, used only if comet_key is not None)')
parser.add_argument('--comet_project_name',  type=str,  dest="comet_project_name",  default=None,  help='Comet experiment name (used only if comet_key is not None)')
parser.add_argument('--comet_exp',  type=str,  dest="comet_exp",  default=None,  help='Comet experiment key to continue logging (used only if comet_key is not None)')

args = parser.parse_args()

style_a = args.style_a
style_b = args.style_b
max_samples_train = args.max_samples_train
max_samples_eval = args.max_samples_eval

if args.lambda_file is not None:
    while not os.path.exists(args.lambda_file):
        time.sleep(60)
    with open(args.lambda_file, 'r') as f:
        args.lambdas = f.read()
    os.remove(args.lambda_file)

hyper_params = {}
print ("Arguments summary: \n ")
for key, value in vars(args).items():
    hyper_params[key] = value
    print (f"\t{key}:\t\t{value}")

# lambdas: cycle-consistency, generator-fooling, disc-fake, disc-real, classifier-guided
lambdas = [float(l) for l in args.lambdas.split('|')]
args.lambdas = lambdas
    
mono_ds_a = MonostyleDataset(dataset_format="line_file",
                            style=style_a,
                            dataset_path=args.path_mono_A,
                            separator='\n',
                            max_dataset_samples=args.max_samples_train)

mono_ds_b = MonostyleDataset(dataset_format="line_file",
                            style=style_b,
                            dataset_path=args.path_mono_B,
                            separator='\n',
                            max_dataset_samples=args.max_samples_train)

if args.nonparal_same_size:
    mono_ds_a_len, mono_ds_b_len = len(mono_ds_a), len(mono_ds_b)
    if mono_ds_a_len > mono_ds_b_len: mono_ds_a.reduce_data(mono_ds_b_len)
    else: mono_ds_b.reduce_data(mono_ds_a_len)

if args.n_references is not None:
    parallel_ds_evalAB = ParallelRefDataset(dataset_format='line_file',
                                            style_src=style_a,
                                            style_ref=style_b,
                                            dataset_path_src=args.path_paral_A_eval,
                                            dataset_path_ref=args.path_paral_eval_ref,
                                            n_ref=args.n_references,
                                            separator_src='\n',
                                            separator_ref='\n',
                                            max_dataset_samples=args.max_samples_eval)
    
    parallel_ds_evalBA = ParallelRefDataset(dataset_format='line_file',
                                            style_src=style_b,
                                            style_ref=style_a,
                                            dataset_path_src=args.path_paral_B_eval,
                                            dataset_path_ref=args.path_paral_eval_ref,
                                            n_ref=args.n_references,
                                            separator_src='\n',
                                            separator_ref='\n',
                                            max_dataset_samples=args.max_samples_eval)
else:
    mono_ds_a_eval = MonostyleDataset(dataset_format="line_file",
                                      style=style_a,
                                      dataset_path=args.path_mono_A_eval,
                                      separator='\n',
                                      max_dataset_samples=args.max_samples_eval)

    mono_ds_b_eval = MonostyleDataset(dataset_format="line_file",
                                      style=style_b,
                                      dataset_path=args.path_mono_B_eval,
                                      separator='\n',
                                      max_dataset_samples=args.max_samples_eval)

print (f"Mono A  : {len(mono_ds_a)}")
print (f"Mono B  : {len(mono_ds_b)}")
if args.n_references is not None:
    print (f"Parallel AB eval: {len(parallel_ds_evalAB)}")
    print (f"Parallel BA eval: {len(parallel_ds_evalBA)}")
else:
    print (f"Mono A eval: {len(mono_ds_a_eval)}")
    print (f"Mono B eval: {len(mono_ds_b_eval)}")
print()


mono_dl_a = DataLoader(mono_ds_a,
                        batch_size=args.batch_size,
                        shuffle=args.shuffle,
                        num_workers=args.num_workers,
                        pin_memory=args.pin_memory)

mono_dl_b = DataLoader(mono_ds_b,
                        batch_size=args.batch_size,
                        shuffle=args.shuffle,
                        num_workers=args.num_workers,
                        pin_memory=args.pin_memory)
del mono_ds_a, mono_ds_b

if args.n_references is not None:
    parallel_dl_evalAB = DataLoader(parallel_ds_evalAB,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=args.pin_memory,
                                    collate_fn=ParallelRefDataset.customCollate)
    
    parallel_dl_evalBA = DataLoader(parallel_ds_evalBA,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=args.pin_memory,
                                    collate_fn=ParallelRefDataset.customCollate)
    del parallel_ds_evalAB, parallel_ds_evalBA
else:
    mono_dl_a_eval = DataLoader(mono_ds_a_eval,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=args.pin_memory)

    mono_dl_b_eval = DataLoader(mono_ds_b_eval,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=args.pin_memory)
    del mono_ds_a_eval, mono_ds_b_eval

if args.n_references is not None:
    print (f"Parallel AB eval (batches): {len(parallel_dl_evalAB)}")
    print (f"Parallel BA eval (batches): {len(parallel_dl_evalBA)}")
else:
    print (f"Mono A eval (batches): {len(mono_dl_a_eval)}")
    print (f"Mono B eval (batches): {len(mono_dl_b_eval)}")


''' 
    ----- ----- ----- ----- ----- ----- ----- -----
              Instantiate Generators       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

if args.from_pretrained is not None:
    G_ab = GeneratorModel(args.generator_model_tag, f'{args.from_pretrained}G_ab/', max_seq_length=args.max_sequence_length)
    G_ba = GeneratorModel(args.generator_model_tag, f'{args.from_pretrained}G_ba/', max_seq_length=args.max_sequence_length)
    print('Generator pretrained models loaded correctly')
else:
    G_ab = GeneratorModel(args.generator_model_tag, max_seq_length=args.max_sequence_length)
    G_ba = GeneratorModel(args.generator_model_tag, max_seq_length=args.max_sequence_length)
    print('Generator pretrained models not loaded - Initial weights will be used')


''' 
    ----- ----- ----- ----- ----- ----- ----- -----
             Instantiate Discriminators       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

if args.from_pretrained is not None:
    D_ab = DiscriminatorModel(args.discriminator_model_tag, f'{args.from_pretrained}D_ab/', max_seq_length=args.max_sequence_length)
    D_ba = DiscriminatorModel(args.discriminator_model_tag, f'{args.from_pretrained}D_ba/', max_seq_length=args.max_sequence_length)
    print('Discriminator pretrained models loaded correctly')
else:
    D_ab = DiscriminatorModel(args.discriminator_model_tag, max_seq_length=args.max_sequence_length)
    D_ba = DiscriminatorModel(args.discriminator_model_tag, max_seq_length=args.max_sequence_length)
    print('Discriminator pretrained models not loaded - Initial weights will be used')


''' 
    ----- ----- ----- ----- ----- ----- ----- -----
             Instantiate Classifier       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

if lambdas[4] != 0:
    Cls = ClassifierModel(args.pretrained_classifier_model, max_seq_length=args.max_sequence_length)
    print('Classifier pretrained model loaded correctly')
else:
    Cls = None


''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                    SETTINGS       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

if args.use_cuda_if_available:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")

cycleGAN = CycleGANModel(G_ab, G_ba, D_ab, D_ba, Cls, device=device)

n_batch_epoch = min(len(mono_dl_a), len(mono_dl_b))
num_training_steps = args.epochs * n_batch_epoch

print(f"Total number of training steps: {num_training_steps}")

warmup_steps = int(0.1*num_training_steps) if args.warmup else 0

optimizer = AdamW(cycleGAN.get_optimizer_parameters(), lr=args.learning_rate)
# scheduler types: ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
lr_scheduler = get_scheduler(args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)
    
start_epoch = 0
current_training_step = 0

if args.from_pretrained is not None:
    checkpoint = torch.load(f"{args.from_pretrained}checkpoint.pth", map_location=torch.device("cpu"))
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch']
    current_training_step = checkpoint['training_step']
    del checkpoint

''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                 COMET LOGGING SETUP
    ----- ----- ----- ----- ----- ----- ----- -----
'''

if args.comet_logging:
    if args.from_pretrained is not None:
        experiment = ExistingExperiment(api_key=args.comet_key, previous_experiment=args.comet_exp)
    else:
        experiment = Experiment(
            api_key=args.comet_key,
            project_name=args.comet_project_name,
            workspace=args.comet_workspace,
        )
    experiment.log_parameters(hyper_params)
else:
    experiment = None

loss_logging = {'Cycle Loss A-B-A':[], 'Loss generator  A-B':[], 'Classifier-guided A-B':[], 'Loss D(A->B)':[],
                'Cycle Loss B-A-B':[], 'Loss generator  B-A':[], 'Classifier-guided B-A':[], 'Loss D(B->A)':[]}
loss_logging['hyper_params'] = hyper_params

''' 
    ----- ----- ----- ----- ----- ----- ----- -----
                    TRAINING LOOP       
    ----- ----- ----- ----- ----- ----- ----- -----
'''

progress_bar = tqdm(range(num_training_steps))
progress_bar.update(current_training_step)

evaluator = Evaluator(cycleGAN, args, experiment)

print('Start training...')
for epoch in range(start_epoch, args.epochs):
    print (f"\nTraining epoch: {epoch}")
    cycleGAN.train() # set training mode

    for unsupervised_a, unsupervised_b in zip(mono_dl_a, mono_dl_b):
        len_a, len_b = len(unsupervised_a), len(unsupervised_b)
        if len_a > len_b: unsupervised_a = unsupervised_a[:len_b]
        else: unsupervised_b = unsupervised_b[:len_a]

        cycleGAN.training_cycle(sentences_a=unsupervised_a,
                                sentences_b=unsupervised_b,
                                lambdas=lambdas,
                                comet_experiment=experiment,
                                loss_logging=loss_logging,
                                training_step=current_training_step)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        current_training_step += 1

        # dummy classification metrics/BERTScore computation to see if it fits in GPU
        if current_training_step==5:
            if args.n_references is None: evaluator.dummy_classif()
            elif args.bertscore: evaluator.dummy_bscore()
        if (args.eval_strategy == "steps" and current_training_step%args.eval_steps==0) or (epoch < args.additional_eval and current_training_step%(n_batch_epoch//2+1)==0):
            if args.n_references is not None:
                evaluator.run_eval_ref(epoch, current_training_step, 'validation', parallel_dl_evalAB, parallel_dl_evalBA)
            else:
                evaluator.run_eval_mono(epoch, current_training_step, 'validation', mono_dl_a_eval, mono_dl_b_eval)
            cycleGAN.train()

    if args.n_references is not None:
        evaluator.run_eval_ref(epoch, current_training_step, 'validation', parallel_dl_evalAB, parallel_dl_evalBA)
    else:
        evaluator.run_eval_mono(epoch, current_training_step, 'validation', mono_dl_a_eval, mono_dl_b_eval)
    if epoch%args.save_steps==0:
        cycleGAN.save_models(f"{args.save_base_folder}epoch_{epoch}/")
        checkpoint = {'epoch':epoch+1, 'training_step':current_training_step, 'optimizer':optimizer.state_dict(), 'lr_scheduler':lr_scheduler.state_dict()}
        torch.save(checkpoint, f"{args.save_base_folder}epoch_{epoch}/checkpoint.pth")
        if epoch > 0 and os.path.exists(f"{args.save_base_folder}epoch_{epoch-1}/checkpoint.pth"):
            os.remove(f"{args.save_base_folder}epoch_{epoch-1}/checkpoint.pth")
        if epoch > 0 and os.path.exists(f"{args.save_base_folder}loss.pickle"):
            os.remove(f"{args.save_base_folder}loss.pickle")
        pickle.dump(loss_logging, open(f"{args.save_base_folder}loss.pickle", 'wb'))
    if args.control_file is not None and os.path.exists(args.control_file):
        with open(args.control_file, 'r') as f:
            if f.read() == 'STOP':
                print(f'STOP command received - Stopped at epoch {epoch}')
                os.remove(args.control_file)
                break
    cycleGAN.train()

print('End training...')
