from comet_ml import Experiment

from utils.utils import *

import argparse, pickle
import numpy as np
from sklearn.metrics import classification_report

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


parser = argparse.ArgumentParser()

parser.add_argument('--max_samples_train', type=int, dest="max_samples_train", default=None, help='Max number of examples to retain from the training set. None for all available examples.')
parser.add_argument('--max_samples_eval',  type=int, dest="max_samples_eval",  default=None, help='Max number of examples to retain from the evaluation set. None for all available examples.')
parser.add_argument('--dataset_path', type=str, dest="dataset_path", help='Path to dataset base folder.')

parser.add_argument('--lowercase', action='store_true', dest="lowercase", default=False, help='Whether to lowercase data.')
parser.add_argument('--max_sequence_length', type=int,  dest="max_sequence_length", default=64, help='Max sequence length')
parser.add_argument('--batch_size', type=int,  dest="batch_size",  default=64,     help='Batch size used during training.')

parser.add_argument('--use_cuda_if_available', action='store_true', dest="use_cuda_if_available", default=False, help='Whether to use GPU if available.')

parser.add_argument('--learning_rate',     type=float, dest="learning_rate",     default=5e-5,     help='Initial learning rate (e.g., 5e-5).')
parser.add_argument('--epochs',            type=int,   dest="epochs",            default=10,       help='The number of training epochs.')
parser.add_argument('--lr_scheduler_type', type=str,   dest="lr_scheduler_type", default="linear", help='The scheduler used for the learning rate management.')

parser.add_argument('--model_tag', type=str, dest="model_tag", help='The tag of the model to train (e.g., "roberta-base").')

parser.add_argument('--save_base_folder', type=str, dest="save_base_folder", help='The folder to use as base path to store model checkpoints')
parser.add_argument('--save_steps',       type=int, dest="save_steps",       help='How many training epochs between two checkpoints.')
parser.add_argument('--eval_strategy',    type=str, dest="eval_strategy",    help='Evaluation strategy for the model (either epochs or steps)')
parser.add_argument('--eval_steps',       type=int, dest="eval_steps",       help='How many training steps between two evaluations.')

parser.add_argument('--comet_logging', action='store_true', dest="comet_logging",   default=False, help='Set flag to enable comet logging')
parser.add_argument('--comet_key',       type=str,  dest="comet_key",       default=None,  help='Comet API key to log some metrics')
parser.add_argument('--comet_workspace', type=str,  dest="comet_workspace", default=None,  help='Comet workspace name (usually username in Comet, used only if comet_key is not None)')
parser.add_argument('--comet_project_name',  type=str,  dest="comet_project_name",  default=None,  help='Comet experiment name (used only if comet_key is not None)')

args = parser.parse_args()

hyper_params = {}
print ("Arguments summary: \n ")
for key,value in vars(args).items():
    hyper_params[key] = value
    print (f"\t{key}:\t\t{value}")

if args.eval_strategy == 'epochs': args.eval_strategy = 'epoch'

if args.use_cuda_if_available:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
else:
    device = torch.device("cpu")


dataset_name = args.dataset_path.split('/')[-2]
output_path = args.save_base_folder + f"classifiers/{dataset_name}/{args.model_tag}_{args.epochs}/"
output_dir, logging_dir = output_path+"checkpoints/", output_path+"logs/"

x_train, y_train = read_data(args.dataset_path, 'train', args.max_samples_train, args.lowercase)
x_eval, y_eval = read_data(args.dataset_path, 'dev', args.max_samples_eval, args.lowercase)
x_test, y_test = read_data(args.dataset_path, 'test', None, args.lowercase)

print (f"Training set: {len(x_train)}")
print (f"Evaluation set: {len(x_eval)}")
print (f"Test set: {len(x_test)}")

n_batch_train = int(np.ceil(len(x_train)/args.batch_size))
print (f"Training set (batches): {n_batch_train}")
print (f"Evaluation set (batches): {int(np.ceil(len(x_eval)/args.batch_size))}")
print (f"Test set (batches): {int(np.ceil(len(x_test)/args.batch_size))}")

model = AutoModelForSequenceClassification.from_pretrained(args.model_tag, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(args.model_tag)
model.to(device)

truncation, padding = 'longest_first', 'max_length'

x_train_tokenized = tokenizer(x_train, truncation=truncation, padding=padding, max_length=args.max_sequence_length)
x_eval_tokenized = tokenizer(x_eval, truncation=truncation, padding=padding, max_length=args.max_sequence_length)
x_test_tokenized = tokenizer(x_test, truncation=truncation, padding=padding, max_length=args.max_sequence_length)

train_ds = ClassifierDataset(x_train_tokenized, y_train)
eval_ds = ClassifierDataset(x_eval_tokenized, y_eval)
test_ds = ClassifierDataset(x_test_tokenized, y_test)

training_args = TrainingArguments(per_device_train_batch_size=args.batch_size,
                                  per_device_eval_batch_size=args.batch_size,
                                  num_train_epochs=args.epochs,
                                  learning_rate=args.learning_rate,
                                  lr_scheduler_type=args.lr_scheduler_type,
                                  warmup_steps=10,
                                  weight_decay=0.01,
                                  output_dir=output_dir,
                                  logging_dir=logging_dir,
                                  logging_strategy='steps',
                                  logging_steps=10,
                                  evaluation_strategy=args.eval_strategy,
                                  eval_steps=args.eval_steps,
                                  save_strategy='epoch',
                                  save_steps=args.save_steps,
                                  load_best_model_at_end=True,
                                  metric_for_best_model='acc',
                                  seed=SEED)

trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_ds,
                  eval_dataset=eval_ds,
                  compute_metrics=classifier_metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)])

print(f"Total number of training steps: {args.epochs * n_batch_train}")

print('Start training...')
trainer.train()
print('End training...')

tokenizer.save_pretrained(f'{output_dir}tokenizer/')

print('Start test...')
predictions = trainer.predict(test_ds)
print('End test...')

y_pred = np.argmax(predictions.predictions, axis=1)
print(classification_report(y_test, y_pred))
test_metrics = classifier_metrics(predictions)

trainer_history = trainer.state.log_history[:-1]
trainer_log = {'hyper_params':hyper_params, 'train_loss':[], 'train_loss_steps':[],
               'eval_loss':[], 'eval_epochs':[], 'eval_steps':[],
               'eval_acc':[], 'eval_prec':[], 'eval_rec':[], 'eval_f1':[]}

for m in trainer_history:
    if 'loss' in m:
        trainer_log['train_loss'].append(m['loss'])
        trainer_log['train_loss_steps'].append(int(m['step']))
    else:
        trainer_log['eval_loss'].append(m['eval_loss'])
        trainer_log['eval_epochs'].append(int(m['epoch']))
        trainer_log['eval_steps'].append(int(m['step']))
        trainer_log['eval_acc'].append(m['eval_acc'])
        trainer_log['eval_prec'].append(m['eval_prec'])
        trainer_log['eval_rec'].append(m['eval_rec'])
        trainer_log['eval_f1'].append(m['eval_f1_macro'])
trainer_log['test_acc'] = test_metrics['acc']
trainer_log['test_prec'] = test_metrics['prec']
trainer_log['test_rec'] = test_metrics['rec']
trainer_log['test_f1'] = test_metrics['f1_macro']
        
pickle.dump(trainer_log, open(f'{output_dir}log.pickle', 'wb'))

if args.comet_logging:
    experiment = Experiment(api_key=args.comet_key,
                            project_name=args.comet_project_name,
                            workspace=args.comet_workspace)
    experiment.log_parameters(hyper_params)

    with experiment.train():
        for loss, step in zip(trainer_log['train_loss'], trainer_log['train_loss_steps']):
            experiment.log_metric(f"Loss classifier", loss, step=step)
    with experiment.validate():
        for loss, epoch, step, acc, prec, rec, f1 in zip(trainer_log['eval_loss'], trainer_log['eval_epochs'], trainer_log['eval_steps'],
                                                         trainer_log['eval_acc'], trainer_log['eval_prec'], trainer_log['eval_rec'], trainer_log['eval_f1']):
            experiment.log_metric(f"Loss classifier", loss, step=step, epoch=epoch)
            experiment.log_metric(f"Accuracy classifier", acc, step=step, epoch=epoch)
            experiment.log_metric(f"Precision classifier", prec, step=step, epoch=epoch)
            experiment.log_metric(f"Recall classifier", rec, step=step, epoch=epoch)
            experiment.log_metric(f"F1 score classifier", f1, step=step, epoch=epoch)
    with experiment.test():
        experiment.log_metric(f"Accuracy classifier", trainer_log['test_acc'], step=0, epoch=0)
        experiment.log_metric(f"Precision classifier", trainer_log['test_prec'], step=0, epoch=0)
        experiment.log_metric(f"Recall classifier", trainer_log['test_rec'], step=0, epoch=0)
        experiment.log_metric(f"F1 score classifier", trainer_log['test_f1'], step=0, epoch=0)
