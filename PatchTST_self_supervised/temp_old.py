

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
from torch.optim import Adam
from src.callback.patch_mask import create_patch

import argparse

parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument('--dset', type=str, default='eeg_time', help='dataset name')
parser.add_argument('--context_points', type=int, default=64, help='sequence length') # default=512
parser.add_argument('--target_points', type=int, default=6, help='forecast horizon')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for DataLoader')
parser.add_argument('--scaler', type=str, default='standard', help='scale the input data')
parser.add_argument('--features', type=str, default='M', help='for multivariate model or univariate model')
# parser.add_argument('--use_time_features', type=int, default=0, help='whether to use time features or not')
# Patch
parser.add_argument('--patch_len', type=int, default=128, help='patch length') # IMPORTANT
parser.add_argument('--stride', type=int, default=128, help='stride between patch')
# RevIN
parser.add_argument('--revin', type=int, default=1, help='reversible instance normalization')
# Model args
parser.add_argument('--n_layers', type=int, default=3, help='number of Transformer layers')
parser.add_argument('--n_heads', type=int, default=16, help='number of Transformer heads')
parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
parser.add_argument('--d_ff', type=int, default=256, help='Tranformer MLP dimension')
parser.add_argument('--dropout', type=float, default=0.2, help='Transformer dropout')
parser.add_argument('--head_dropout', type=float, default=0, help='head dropout')
# Optimization args
parser.add_argument('--n_epochs', type=int, default=100, help='number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
# model id to keep track of the number of models saved
parser.add_argument('--model_id', type=int, default=1, help='id of the saved model')
parser.add_argument('--model_type', type=str, default='based_model', help='for multivariate model or univariate model')
# training
parser.add_argument('--is_train', type=int, default=1, help='training the model')
# adding vitaldb dataset args
parser.add_argument('--segment_sec', type=int, default=20, help='segment length in seconds')
parser.add_argument('--eeg_rate', type=int, default=128, help='EEG sampling rate')
parser.add_argument('--emg_rate', type=int, default=1, help='EMG sampling rate')
parser.add_argument('--stride_sec', type=int, default=20, help='stride length in seconds')
parser.add_argument('--mode', type=str, default='alltrain', help='mode of the dataset, pretrain, alltrain or finetune')
# adding some new args
parser.add_argument('--c_in', type=int, default=1, help='number of input channels')
parser.add_argument('--target_dim', type=int, default=1, help='number of output channels')
parser.add_argument('--num_patch', type=int, default=20, help='number of patches') # default=4


args = parser.parse_args()
print('args:', args)
args.save_model_name = 'patchtst_supervised'+'_cw'+str(args.context_points)+'_tw'+str(args.target_points) + '_patch'+str(args.patch_len) + '_stride'+str(args.stride)+'_epochs'+str(args.n_epochs) + '_model' + str(args.model_id)
args.save_path = 'saved_models/' + args.dset + '/patchtst_supervised/' + args.model_type + '/'
if not os.path.exists(args.save_path): os.makedirs(args.save_path)


def get_model(args):
    """
    c_in: number of input variables
    """
    # # get number of patches
    # num_patch = (max(args.context_points, args.patch_len)-args.patch_len) // args.stride + 1    
    # print('number of patches:', num_patch)
    
    # get model
    model = PatchTST(c_in=args.c_in,
                target_dim=args.target_points,
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
    # get dataloader
    dls = get_dls(args)    
    # model = get_model(dls.vars, args)
    model = get_model(args)
    # get loss
    # loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    # get callbacks
    # cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs = []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    # define learner
    learn = Learner(dls, model, loss_func, cbs=cbs)                        
    # fit the data to the model
    return learn.lr_finder()


def train_func(lr=args.lr):
    # get dataloader
    dls = get_dls(args)
    # print('in out', dls.vars, dls.c, dls.len)
    
    # get model
    # model = get_model(dls.vars, args)
    model = get_model(args)

    # get loss
    # loss_func = torch.nn.MSELoss(reduction='mean')
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')

    # get callbacks
    # cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs = []
    cbs += [
         PatchCB(patch_len=args.patch_len, stride=args.stride),
         SaveModelCB(monitor='valid_loss', fname=args.save_model_name, 
                     path=args.save_path )
        ]

    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=lr, 
                        cbs=cbs,
                        # metrics=[mse]
                        metrics=[accuracy]
                        )
                        
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=lr, pct_start=0.2)


def temp_train():
    dls = get_dls(args)
    train_dataloader = dls.train
    val_dataloader = dls.valid
    model = get_model(args)

    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(100):

        # training phase
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        model.train()
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs, _ = create_patch(inputs, args.patch_len, args.stride)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            # print(f"train_data: labels: {labels}, predicted: {predicted}")

        train_loss /= len(val_dataloader)
        train_accuracy = train_correct / train_total
        
        
        # validation phase
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_dataloader):
                inputs, _ = create_patch(inputs, args.patch_len, args.stride)
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                valid_loss += loss.item()
                
                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()

        valid_loss /= len(val_dataloader)
        valid_accuracy = valid_correct / valid_total
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}')




def test_func():
    weight_path = args.save_path + args.save_model_name + '.pth'
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args)
    #model = torch.load(weight_path)
    # get callbacks
    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
    learn = Learner(dls, model,cbs=cbs)
    out  = learn.test(dls.test, weight_path=weight_path, scores=[mse,mae])         # out: a list of [pred, targ, score_values]
    return out


if __name__ == '__main__':

    # if args.is_train:   # training mode
    #     # suggested_lr = find_lr()
    #     suggested_lr = args.lr  # Use the default learning rate for supervised training
    #     print('suggested lr:', suggested_lr)
    #     train_func(suggested_lr)
    # else:   # testing mode
    #     out = test_func()
    #     print('score:', out[2])
    #     print('shape:', out[0].shape)
   # 
    # print('----------- Complete! -----------')
    temp_train()


