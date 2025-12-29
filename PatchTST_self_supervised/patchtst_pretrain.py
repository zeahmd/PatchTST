

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *


import argparse
from pprint import pprint
from functools import partial
import optuna
from optuna.trial import TrialState

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dataset', type=str, default='eeg_freq', help='dataset name')
parser.add_argument('--input_channels', type=int, default=1, help='number of input channels')
parser.add_argument('--num_classes', type=int, default=4, help='number of output channels')
parser.add_argument('--num_patch', type=int, default=30, help='number of patches')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for DataLoader')
parser.add_argument('--debug', type=int, default=1, help='whether to use debug mode with small dataset')
parser.add_argument('--use_weighted_sampler', type=int, default=0, help='whether to use weighted sampler for imbalanced data')
# adding vitaldb dataset args
parser.add_argument('--segment_sec', type=int, default=30, help='segment length in seconds')
parser.add_argument('--eeg_rate', type=int, default=128, help='EEG sampling rate')
parser.add_argument('--emg_rate', type=int, default=1, help='EMG sampling rate')
parser.add_argument('--stride_sec', type=int, default=30, help='stride length in seconds')
parser.add_argument('--mode', type=str, default='pretrain', help='mode of the dataset, pretrain, alltrain or finetune')
# Patch
parser.add_argument('--patch_len', type=int, default=65, help='patch length') # default=512
parser.add_argument('--stride', type=int, default=65, help='stride between patch') # default=256
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
parser.add_argument('--use_emg', type=int, default=0, help='whether to use emg as additional input feature')
# Pretrain mask
parser.add_argument('--mask_ratio', type=float, default=0.4, help='masking ratio for the input')
# Optimization args
parser.add_argument('--n_epochs_pretrain', type=int, default=3, help='number of pre-training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for optimizer')
# model id to keep track of the number of models saved
parser.add_argument('--pretrained_model_id', type=int, default=1, help='id of the saved pretrained model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# training
parser.add_argument('--save_path', type=str, default='saved_models/', help='path to save the model')
parser.add_argument('--save_model_name', type=str, default='unnammed_model', help='name of the saved model')
# Hyperparameter optimization
parser.add_argument('--is_hyperopt', type=int, default=1, help='whether to do hyperparameter optimization')
parser.add_argument('--n_trials', type=int, default=5, help='number of hyperparameter optimization trials')

args = parser.parse_args()
# args.save_model_name = 'patchtst_supervised'+'_batch'+str(args.batch_size)+'_patch_len'+str(args.patch_len) + '_num_patch'+str(args.num_patch) + '_mode'+str(args.mode)+'_epochs'+str(args.epochs) + '_model' + str(args.model_id)
# args.save_path = 'saved_models/' + args.dataset + '/patchtst_supervised/' + args.model_type + '/'
########### calculate num_patches ###########
if args.dataset == 'eeg_time':
    patch_len_sec = args.patch_len / args.eeg_rate
    num_patches = args.segment_sec // patch_len_sec
    args.num_patch = int(num_patches)
    assert (args.patch_len == 128 and args.stride == 128), "For eeg_time dataset, patch_len and stride must be 128"
if args.dataset == 'eeg_freq':
    patch_len_sec = args.patch_len / 65
    num_patches = args.segment_sec // patch_len_sec
    args.num_patch = int(num_patches)
    assert (args.patch_len == 65 and args.stride == 65), "For eeg_freq dataset, patch_len and stride must be 65"
#############################################
assert args.num_patch == 30, "num_patch != 30, please recalculate based on segment_sec and patch_len"
# args.save_model_name = f"patchtst_pretraining_{args.dataset}_nlayers{args.n_layers}_segment_sec{args.segment_sec}_patch_len{args.patch_len}_num_patch{args.num_patch}_epochs{args.n_epochs_pretrain}_mask{args.mask_ratio}"
# args.save_path = f"saved_models/patchtst_self_supervised/{args.dataset}/eeg_time_pretrain_exps"
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
pprint(vars(args))

# get available GPU devide
# set_device()


# def get_model(c_in, args):
def get_model(args):
    """
    c_in: number of variables
    """
    # get number of patches
    # num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    # print('number of patches:', num_patch)
    
    # get model
    model = PatchTST(c_in=args.input_channels,
                target_dim=args.num_classes,
                patch_len=args.patch_len,
                stride=args.stride,
                num_patch=args.num_patch,
                use_emg=args.use_emg,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                d_model=args.d_model,
                shared_embedding=True,
                d_ff=args.d_ff,                        
                dropout=args.dropout,
                head_dropout=args.head_dropout,
                act='relu',
                head_type='pretrain',
                res_attention=False
                )        
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_lr():
    # get dataloader
    dls = get_dls(args)    
    # model = get_model(dls.vars, args)
    model = get_model(args)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    # cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs = []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]
        
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=args.lr, 
                        cbs=cbs,
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def hyperopt_func(dls, trial):
    args.lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    args.n_layers = trial.suggest_categorical('n_layers', [3, 6])
    args.n_heads = trial.suggest_categorical('n_heads', [8, 16, 32])
    args.d_model = trial.suggest_categorical('d_model', [128, 256])
    args.d_ff = trial.suggest_categorical('d_ff', [256, 512])
    args.dropout = trial.suggest_float('dropout', 0.2, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 0.05, log=True)
    print(f'Running trial {trial._trial_id} for {args.n_epochs_pretrain} epochs!')

    # get model
    model = get_model(args)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = []
    cbs += [
         PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name,          
                        path=args.save_path)
        ]
    # define learner
    learn = Learner(args, dls, model, 
                        loss_func, 
                        lr=args.lr,
                        l2_reg=weight_decay,
                        cbs=cbs,
                        # metrics=[mse]
                        )                        
    # fit the data to the model
    last_epoch_val_loss = learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=args.lr)
    return last_epoch_val_loss


def pretrain_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    # get model     
    # model = get_model(dls.vars, args)
    model = get_model(args)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    # cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    # cbs = [RevInCB(args.c_in, denorm=False)] if args.revin else [] # we don't need revin for vitaldb
    cbs = []
    cbs += [
         PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name,          
                        path=args.save_path),
        SaveHistoryCB(path=args.save_path, fname=args.save_model_name)
        ]
    # define learner
    learn = Learner(args, dls, model, 
                        loss_func, 
                        lr=lr,
                        l2_reg=args.weight_decay,
                        cbs=cbs,
                        # metrics=[mse]
                        )                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=lr)

    # train_loss = learn.recorder['train_loss']
    # valid_loss = learn.recorder['valid_loss']
    # df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    # df.to_csv(args.save_path + args.save_model_name + '_losses.csv', float_format='%.6f', index=False)


if __name__ == '__main__':
    if args.is_hyperopt:
        # Hyperparameter optimization
        print("Starting hyperparameter optimization...")
        dls = get_dls(args)
        hyperopt_func = partial(hyperopt_func, dls)

        print(f"Running {args.n_trials} trials...")
        study = optuna.create_study(direction='minimize')
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
        args.dset = args.dataset
        # suggested_lr = find_lr()
        suggested_lr = args.lr
        print('suggested_lr', suggested_lr)
        # Pretrain
        # suggested_lr = args.lr  # Use the default learning rate for pretraining
        pretrain_func(suggested_lr)
        print('pretraining completed')
    

    

