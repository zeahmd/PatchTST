

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner
from src.callback.core import *
from src.callback.tracking import *
from src.callback.scheduler import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from datautils import get_dls
from src.basics import default_device


import argparse
from pprint import pprint
from functools import partial
import optuna
from optuna.trial import TrialState

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dataset', type=str, default='eeg_time_freq', help='dataset name')
parser.add_argument('--input_channels', type=int, default=1, help='number of input channels')
parser.add_argument('--num_classes', type=int, default=4, help='number of output channels')
parser.add_argument('--num_patch', type=int, default=30, help='number of patches')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for DataLoader')
parser.add_argument('--debug', type=int, default=1, help='whether to use debug mode')
# adding vitaldb dataset args
parser.add_argument('--segment_sec', type=int, default=30, help='segment length in seconds')
parser.add_argument('--eeg_rate', type=int, default=128, help='EEG sampling rate')
parser.add_argument('--emg_rate', type=int, default=1, help='EMG sampling rate')
parser.add_argument('--stride_sec', type=int, default=30, help='stride length in seconds')
parser.add_argument('--mode', type=str, default='finetune', help='mode of the dataset, pretrain, alltrain or finetune')
# Patch
parser.add_argument('--time_patch_len', type=int, default=128, help='patch length')
parser.add_argument('--freq_patch_len', type=int, default=65, help='frequency patch length')
parser.add_argument('--time_stride', type=int, default=128, help='stride between patch')
parser.add_argument('--freq_stride', type=int, default=65, help='stride between frequency patch')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--use_emg', type=int, default=1, help='whether to use emg prediction head')
# Optimization args
parser.add_argument('--epochs', type=int, default=3, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for optimizer')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# training
parser.add_argument('--is_train', type=int, default=0, help='training the model')
parser.add_argument('--save_path', type=str, default='saved_models/', help='path to save the model')
parser.add_argument('--save_model_name', type=str, default='unnammed_model', help='name of the saved model')
# Hyperparameter optimization
parser.add_argument('--is_hyperopt', type=int, default=1, help='whether to do hyperparameter optimization')
parser.add_argument('--n_trials', type=int, default=5, help='number of hyperparameter optimization trials')


args = parser.parse_args()
# args.save_model_name = 'patchtst_supervised'+'_batch'+str(args.batch_size)+'_patch_len'+str(args.patch_len) + '_num_patch'+str(args.num_patch) + '_mode'+str(args.mode)+'_epochs'+str(args.epochs) + '_model' + str(args.model_id)
# args.save_path = 'saved_models/' + args.dataset + '/patchtst_supervised/' + args.model_type + '/'
########### calculate num_patches ###########
patch_len_sec = args.time_patch_len / args.eeg_rate
num_patches = args.segment_sec // patch_len_sec
args.num_patch = int(num_patches)
#############################################
assert args.num_patch == 30, f'num_patch calculated: {args.num_patch}, expected 30'
# args.save_model_name = f"patchtst_supervised_{args.dataset}_segment_sec{args.segment_sec}_patch_len{args.time_patch_len}_num_patch{args.num_patch}_epochs{args.epochs}"
# args.save_path = f"saved_models/patchtst_supervised/{args.dataset}/patch_size_exps"
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
pprint(vars(args))

def get_model(args):
    model = PatchTST(c_in=args.input_channels,
                target_dim=args.num_classes,
                time_patch_len=args.time_patch_len,
                freq_patch_len=args.freq_patch_len,
                time_stride=args.time_stride,
                freq_stride=args.freq_stride,
                use_emg=args.use_emg,
                num_patch=args.num_patch,                
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                # head_type='regression',
                head_type='classification',
                res_attention=False
                )    
    return model


def find_lr():
    # get dataloader and model
    dls = get_dls(args)    
    model = get_model(args)
    # get loss
    # loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    cbs = []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    # define learner
    learn = Learner(dls, model, loss_func, cbs=cbs)                        
    # fit the data to the model
    return learn.lr_finder()


def train_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    train_label_dist = dls.get_label_distribution('train')
    train_label_dist = [v for k, v in sorted(train_label_dist.items(), key=lambda item: item[0])]
    # these class weights are computed as in scikit-learn compute_class_weight("balanced", ...)
    # it makes sure that mean(weights) = 1.0 (and not the sum)
    train_class_weights = 1.0 / torch.tensor(train_label_dist, dtype=torch.float)
    train_class_weights = train_class_weights / train_class_weights.sum() * len(train_label_dist)
    # print('in out', dls.vars, dls.c, dls.len)
    
    # get model
    model = get_model(args)

    # get loss
    # loss_func = torch.nn.MSELoss(reduction='mean')
    # loss_func = torch.nn.CrossEntropyLoss(
    #     weight=train_class_weights.to(default_device(use_cuda=True)),
    #     reduction='mean'
    # )
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

    # get callbacks
    cbs = []
    cbs += [
         PatchCB(time_patch_len=args.time_patch_len, freq_patch_len=args.freq_patch_len, time_stride=args.time_stride, freq_stride=args.freq_stride),
         SaveModelCB(monitor='valid_f1_score', fname=args.save_model_name, 
                     path=args.save_path),
        SaveHistoryCB(path=args.save_path, fname=args.save_model_name)
        ]

    # define learner
    learn = Learner(args, dls,
                        model, 
                        loss_func, 
                        lr=lr,
                        l2_reg=args.weight_decay,
                        cbs=cbs,
                        # metrics=[mse]
                        metrics=[
                                accuracy,
                                precision,
                                recall,
                                f1_score,
                                auroc,
                                conf_mat
                            ]
                        )
                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.epochs, lr_max=lr, pct_start=0.2)


def hyperopt_func(dls, trial):
    args.lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    args.n_layers = trial.suggest_categorical('n_layers', [3, 6])
    args.n_heads = trial.suggest_categorical('n_heads', [8, 16, 32])
    args.d_model = trial.suggest_categorical('d_model', [128, 256])
    args.d_ff = trial.suggest_categorical('d_ff', [256, 512])
    args.dropout = trial.suggest_float('dropout', 0.2, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 0.1, log=True)
    print(f'Running trial {trial._trial_id} for {args.epochs} epochs!')

    # get model
    model = get_model(args)
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    # get callbacks
    cbs = []
    cbs += [
         PatchCB(time_patch_len=args.time_patch_len, freq_patch_len=args.freq_patch_len, time_stride=args.time_stride, freq_stride=args.freq_stride),
         SaveModelCB(monitor='valid_f1_score', fname=args.save_model_name, 
                     path=args.save_path),
        SaveHistoryCB(path=args.save_path, fname=args.save_model_name)
        ]

    # define learner
    learn = Learner(args, dls, model, 
                        loss_func, 
                        lr=args.lr,
                        l2_reg=weight_decay, 
                        cbs=cbs,
                        # metrics=[mse]
                        metrics=[
                                accuracy,
                                precision,
                                recall,
                                f1_score,
                                auroc,
                                conf_mat
                            ],
                        )
                        
    # fit the data to the model
    last_epoch_val_f1_score =learn.fit_one_cycle(n_epochs=args.epochs, lr_max=args.lr, pct_start=0.2)
    return last_epoch_val_f1_score


def test_func(dls):
    # weight_path = args.save_path + args.save_model_name + '.pth'
    model = get_model(args)
    # model = torch.load(args.weight_path)
    # get callbacks
    # cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs = []
    cbs += [PatchCB(time_patch_len=args.time_patch_len, freq_patch_len=args.freq_patch_len, time_stride=args.time_stride, freq_stride=args.freq_stride),]
    learn = Learner(args, dls, model,cbs=cbs)
    out  = learn.test(dls.test, weight_path=args.weight_path, scores=[
        accuracy,
        precision,
        recall,
        f1_score,
        auroc,
        conf_mat
    ])         # out: a list of [pred, targ, score_values]
    return out


if __name__ == '__main__':

    if args.is_train:   # training mode
        if args.is_hyperopt:
            # Hyperparameter optimization
            print("Starting hyperparameter optimization...")
            dls = get_dls(args)
            hyperopt_func = partial(hyperopt_func, dls)

            print(f"Running {args.n_trials} trials...")
            study = optuna.create_study(direction='maximize')
            study.optimize(hyperopt_func, n_trials=args.n_trials)

            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))

            print("Best trial:")
            trial = study.best_trial

            print("  Value: ", trial.value)

            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
        else:
            # suggested_lr = find_lr()
            # suggested_lr = 0.004037017258596558
            suggested_lr = args.lr  # Use the default learning rate for supervised training
            print('suggested lr:', suggested_lr)
            train_func(suggested_lr)
    else:   # testing mode
        # load model config from slurm output
        args.slurm_outfile = "logs/eeg_time_frequency_baseline_3layers_emg_only_1475232.out"
        import re
        import ast
        with open(args.slurm_outfile, 'r') as f:
            content = f.read()

        pattern = r'\{[^{}]*\}'
        matches = re.findall(pattern, content)
        matches = list(filter(lambda x: 'batch_size' in x, matches))

        for i in range(len(matches)):
            train_args = ast.literal_eval(matches[i])
            for key, value in train_args.items():
                setattr(args, key, value)
            args.weight_path = train_args['save_path']+ "/" + train_args['save_model_name'] + '.pth'
            args.is_train = 0

            # IMPORTANT: args.debug = 0
            args.debug = 0
            # get dataloader
            dls = get_dls(args)
            print(f"Testing dataset={args.dataset}, use_emg={train_args['use_emg']}, weights_path={args.weight_path}")
            out = test_func(dls)
            print('score:', out[2])
            print('shape:', out[0].shape)
   
    print('----------- Complete! -----------')


