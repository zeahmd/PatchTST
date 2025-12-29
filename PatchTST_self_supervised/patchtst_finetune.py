

import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.patchTST import PatchTST
from src.learner import Learner, transfer_weights
from src.callback.core import *
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
# Pretraining and Finetuning
parser.add_argument('--is_finetune', type=int, default=0, help='do finetuning or not')
parser.add_argument('--is_linear_probe', type=int, default=0, help='if linear_probe: only finetune the last layer')
# Dataset and dataloader
parser.add_argument('--dataset', type=str, default='eeg_time', help='dataset name')
parser.add_argument('--input_channels', type=int, default=1, help='number of input channels')
parser.add_argument('--num_classes', type=int, default=4, help='number of output channels')
parser.add_argument('--num_patch', type=int, default=30, help='number of patches')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for DataLoader')
parser.add_argument('--debug', type=int, default=0, help='whether to use debug mode with small dataset')
parser.add_argument('--use_weighted_sampler', type=int, default=1, help='whether to use weighted sampler for imbalanced data')
# adding vitaldb dataset args
parser.add_argument('--segment_sec', type=int, default=30, help='segment length in seconds')
parser.add_argument('--eeg_rate', type=int, default=128, help='EEG sampling rate')
parser.add_argument('--emg_rate', type=int, default=1, help='EMG sampling rate')
parser.add_argument('--stride_sec', type=int, default=30, help='stride length in seconds')
parser.add_argument('--mode', type=str, default='finetune', help='mode of the dataset, pretrain, alltrain or finetune')
# Patch
parser.add_argument('--patch_len', type=int, default=128, help='patch length')
parser.add_argument('--stride', type=int, default=128, help='stride between patch')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0.2, help='head dropout')
parser.add_argument('--use_emg', type=int, default=0, help='whether to use emg as additional input feature')
# Optimization args
parser.add_argument('--epochs', type=int, default=20, help='number of finetuning epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay for optimizer')
# Pretrained model name
parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained model name')
# model id to keep track of the number of models saved
parser.add_argument('--finetuned_model_id', type=int, default=1, help='id of the saved finetuned model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# testing
parser.add_argument('--weight_path', type=str, default='', help='path to the saved model for testing')
# Hyperparameter optimization
parser.add_argument('--is_hyperopt', type=int, default=1, help='whether to do hyperparameter optimization')
parser.add_argument('--n_trials', type=int, default=5, help='number of hyperparameter optimization trials')
# model saving path and name
parser.add_argument('--save_path', type=str, default='saved_models/', help='path to save the model')
parser.add_argument('--save_model_name', type=str, default='unnammed_model', help='name of the saved model')


args = parser.parse_args()
########## Explitcity setting finetuned model name ##########
# args.is_finetune = True
# args.pretrained_model = "saved_models/patchtst_self_supervised/eeg_time/eeg_time_pretrain_exps/patchtst_pretraining_eeg_time_nlayers3_segment_sec30_patch_len128_num_patch30_epochs200_mask0.4.pth"
#############################################################
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
    assert (args.patch_len % 65 == 0 and args.stride % 65 == 0), "For eeg_freq dataset, patch_len and stride must be 65, 130, 195, etc."
#############################################
# args.save_path = f"saved_models/finetuned/{args.dataset}/eeg_time_finetune_exps"
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
pprint(vars(args))

# if args.is_finetune: args.save_model_name = f"patchtst_finetuned_{args.dataset}_nlayers{args.n_layers}_segment_sec{args.segment_sec}_patch_len{args.patch_len}_num_patch{args.num_patch}_finetune_epochs{args.epochs}"
# elif args.is_linear_probe: args.save_model_name = f"patchtst_finetune-linear-probe_{args.dataset}_nlayers{args.n_layers}_segment_sec{args.segment_sec}_patch_len{args.patch_len}_num_patch{args.num_patch}_finetune_epochs{args.epochs}"
# else: args.save_model_name = f"patchtst_finetuned_{args.dataset}_nlayers{args.n_layers}_segment_sec{args.segment_sec}_patch_len{args.patch_len}_num_patch{args.num_patch}_finetune_epochs{args.epochs}"

# get available GPU devide
# set_device()

def get_model(args, weight_path=None):
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
                head_type='classification',
                res_attention=False
                )    
    if weight_path: model = transfer_weights(weight_path, model)
    # print out the model size
    print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model



def find_lr(head_type):
    # get dataloader
    dls = get_dls(args)    
    model = get_model(dls.vars, args, head_type)
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        
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


def save_recorders(learn):
    train_loss = learn.recorder['train_loss']
    valid_loss = learn.recorder['valid_loss']
    df = pd.DataFrame(data={'train_loss': train_loss, 'valid_loss': valid_loss})
    df.to_csv(args.save_path + args.save_model_name + '_losses.csv', float_format='%.6f', index=False)


def finetune_func(lr=args.lr):
    print('end-to-end finetuning')
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(args)
    # transfer weight
    # weight_path = args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')   
    # get callbacks
    cbs = []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_f1_score', fname=args.save_model_name, path=args.save_path),
         SaveHistoryCB(path=args.save_path, fname=args.save_model_name)
        ]
    # define learner
    learn = Learner(args, dls, model, 
                        loss_func, 
                        lr=lr, 
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
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    learn.fine_tune(n_epochs=args.epochs, base_lr=lr, freeze_epochs=10)
    save_recorders(learn)


def hyperopt_func(dls, trial):
    args.lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    args.dropout = trial.suggest_float('dropout', 0.2, 0.5)
    args.head_dropout = trial.suggest_float('head_dropout', 0.2, 0.5)
    weight_decay = trial.suggest_float('weight_decay', 1e-3, 0.1, log=True)
    print(f'Running trial {trial._trial_id} for {args.epochs} epochs!')

    # get model 
    model = get_model(args)
    # transfer weight
    # weight_path = args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')   
    # get callbacks
    cbs = []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_f1_score', fname=args.save_model_name, path=args.save_path),
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
                            ]
                        )                            
    # fit the data to the model
    #learn.fit_one_cycle(n_epochs=args.n_epochs_finetune, lr_max=lr)
    last_epoch_val_f1_score = learn.fine_tune(n_epochs=args.epochs, base_lr=args.lr, freeze_epochs=10)
    return last_epoch_val_f1_score


def linear_probe_func(lr=args.lr):
    print('linear probing')
    # get dataloader
    dls = get_dls(args)
    # get model 
    model = get_model(args)
    # transfer weight
    # weight_path = args.save_path + args.pretrained_model + '.pth'
    model = transfer_weights(args.pretrained_model, model)
    # get loss
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')    
    # get callbacks
    cbs = []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_f1_score', fname=args.save_model_name, path=args.save_path)
        ]
    # define learner
    learn = Learner(args, dls, model, 
                        loss_func, 
                        lr=lr, 
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
    learn.linear_probe(n_epochs=args.epochs, base_lr=lr)
    save_recorders(learn)


def test_func(dls):
    # weight_path = args.save_path + args.save_model_name + '.pth'
    model = get_model(args)
    # model = torch.load(args.weight_path)
    # get callbacks
    # cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs = []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(args, dls, model, cbs=cbs)
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
        
    if args.is_finetune:
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
            # Finetune
            # suggested_lr = find_lr(head_type='prediction')  
            suggested_lr = args.lr      
            finetune_func(suggested_lr)        
            print('finetune completed')
            # Test
            # out = test_func(args.save_path+args.save_finetuned_model)         
            # print('----------- Complete! -----------')

    elif args.is_linear_probe:
        # args.dset = args.dset_finetune
        # Finetune
        # suggested_lr = find_lr(head_type='prediction')        
        suggested_lr = 1e-4      
        linear_probe_func(suggested_lr)        
        print('finetune completed')
        # Test
        out = test_func(args.save_path+args.save_finetuned_model)        
        print('----------- Complete! -----------')

    else:
        # args.dset = args.dset_finetune
        # weight_path = args.save_path+args.dset_finetune+'_patchtst_finetuned'+suffix_name
        # # Test
        # out = test_func(weight_path)        
        # print('----------- Complete! -----------')

        # load model config from slurm output
        args.slurm_outfile = "logs/eeg_freq_finetune_train_1475506.out"
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


