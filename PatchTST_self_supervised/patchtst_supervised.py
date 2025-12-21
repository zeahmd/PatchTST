

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

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dataset', type=str, default='eeg_time', help='dataset name')
parser.add_argument('--input_channels', type=int, default=1, help='number of input channels')
parser.add_argument('--num_classes', type=int, default=4, help='number of output channels')
parser.add_argument('--num_patch', type=int, default=20, help='number of patches')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers for DataLoader')
# adding vitaldb dataset args
parser.add_argument('--segment_sec', type=int, default=20, help='segment length in seconds')
parser.add_argument('--eeg_rate', type=int, default=128, help='EEG sampling rate')
parser.add_argument('--emg_rate', type=int, default=1, help='EMG sampling rate')
parser.add_argument('--stride_sec', type=int, default=20, help='stride length in seconds')
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
parser.add_argument('--head_dropout', type=float, default=0.1, help='head dropout')
# Optimization args
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# training
parser.add_argument('--is_train', type=int, default=1, help='training the model')



args = parser.parse_args()
# args.save_model_name = 'patchtst_supervised'+'_batch'+str(args.batch_size)+'_patch_len'+str(args.patch_len) + '_num_patch'+str(args.num_patch) + '_mode'+str(args.mode)+'_epochs'+str(args.epochs) + '_model' + str(args.model_id)
# args.save_path = 'saved_models/' + args.dataset + '/patchtst_supervised/' + args.model_type + '/'
########### calculate num_patches ###########
patch_len_sec = args.patch_len / args.eeg_rate
num_patches = args.segment_sec // patch_len_sec
args.num_patch = int(num_patches)
#############################################
args.save_model_name = f"patchtst_supervised_{args.dataset}_nlayers{args.n_layers}_segment_sec{args.segment_sec}_patch_len{args.patch_len}_num_patch{args.num_patch}_epochs{args.epochs}"
args.save_path = f"saved_models/patchtst_supervised/{args.dataset}/weighted_sampler_exps"
if not os.path.exists(args.save_path): os.makedirs(args.save_path)
pprint(vars(args))

def get_model(args):
    model = PatchTST(c_in=args.input_channels,
                target_dim=args.num_classes,
                patch_len=args.patch_len,
                stride=args.stride,
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
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    # get callbacks
    cbs = []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name, 
                     path=args.save_path),
        SaveHistoryCB(path=args.save_path, fname=args.save_model_name)
        ]

    # define learner
    learn = Learner(dls, model, 
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
    learn.fit_one_cycle(n_epochs=args.epochs, lr_max=lr, pct_start=0.2)



def test_func():
    # weight_path = args.save_path + args.save_model_name + '.pth'
    # get dataloader
    dls = get_dls(args)
    model = get_model(args)
    # model = torch.load(args.weight_path)
    # get callbacks
    # cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs = []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)
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
        # suggested_lr = find_lr()
        # suggested_lr = 0.004037017258596558
        suggested_lr = args.lr  # Use the default learning rate for supervised training
        print('suggested lr:', suggested_lr)
        train_func(suggested_lr)
    else:   # testing mode
        args.weight_path = "saved_models/eeg_time/patchtst_supervised/based_model/patchtst_supervised_batch512_patch_len128_num_patch20_modealltrain_epochs1_model1.pth"
        out = test_func()
        print('score:', out[2])
        print('shape:', out[0].shape)
   
    print('----------- Complete! -----------')


