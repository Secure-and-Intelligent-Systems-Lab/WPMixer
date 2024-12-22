import os
import torch
import argparse
import optuna

torch.set_printoptions(precision = 10)
from exp.exp_main import Exp_Main
import numpy as np
import random 
from utils.tools import dotdict
import gc
from utils.Tuner import Tuner
from utils.output_database import Output_database
from utils.tools import set_random_seed
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = '[WPMixer] Long Sequences Forecasting')
    
    ''' frequent changing hy.params '''
    parser.add_argument('--model', type = str, required = False, choices = ['WPMixer'], default = 'WPMixer',help = 'model of experiment')
    parser.add_argument('--task_name', type = str, required = False, choices = ['long_term_forecast'], default = 'long_term_forecast')
    parser.add_argument('--data', type = str, default = 'ETTh1', choices = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'Weather', 'Traffic'], help = 'dataset')
    parser.add_argument('--use_hyperParam_optim', action = 'store_true', default = False, help = 'True: HyperParameters optimization using optuna, False: no optimization')
    parser.add_argument('--no_decomposition', action = 'store_true', default = False, help = 'whether to use wavelet decomposition')
    parser.add_argument('--use_multi_gpu', action = 'store_true', help = 'use multiple gpus', default = False)
    parser.add_argument('--n_jobs', type = int, required = False, choices = [1, 2, 3, 4], default = 1, help = 'number_of_jobs for optuna')
    parser.add_argument('--seed', type = int, required = False, default = 42, help = 'random seed')
    
    ''' Model Parameters '''
    parser.add_argument('--seq_len', type = int, default = 512, help = 'length of the look back window')
    parser.add_argument('--pred_len', type = int, default = 96, help = 'prediction length')
    parser.add_argument('--d_model', type = int, default = 256, help = 'embedding dimension')
    parser.add_argument('--tfactor', type = int, default = 5, help = 'expansion factor in the patch mixer')
    parser.add_argument('--dfactor', type = int, default = 5, help = 'expansion factor in the embedding mixer')
    parser.add_argument('--wavelet', type = str, default = 'db2', help = 'wavelet type for wavelet transform')
    parser.add_argument('--level', type = int, default = 1, help = 'level for multi-level wavelet decomposition')
    parser.add_argument('--patch_len', type = int, default = 16, help = 'Patch size')
    parser.add_argument('--stride', type = int, default = 8, help = 'Stride')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'batch size')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'initial learning rate')    
    parser.add_argument('--dropout', type = float, default = 0.05, help = 'dropout for mixer')
    parser.add_argument('--embedding_dropout', type = float, default = 0.05, help = 'dropout for embedding layer')
    parser.add_argument('--weight_decay', type = float, default = 0.00, help = 'pytorch weight decay factor')
    parser.add_argument('--patience', type = int, default = 10, help = 'patience')
    parser.add_argument('--train_epochs', type = int, default = 10, help = 'train epochs')
    
    ''' Infrequent chaning parameters: Some of these has not used in our model '''
    parser.add_argument('--label_len', type = int, default = 0, help = 'label length')
    parser.add_argument('--seasonal_patterns', type = str, default = 'Monthly', help = 'subset for M4')
    parser.add_argument('--features', type = str, default = 'M', help = 'forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type = str, default = 'OT', help = 'target feature in S or MS task')
    parser.add_argument('--freq', type = str, default = 'h', help = 'freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type = str, default = './checkpoints/', help = 'location of model checkpoints')
    parser.add_argument('--cols', type = str, nargs = '+', default = None, help = 'certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'data loader num workers')
    parser.add_argument('--itr', type = int, default = 1, help = 'experiments times')
    parser.add_argument('--lradj', type = str, default = 'type3', help = 'adjust learning rate')
    parser.add_argument('--use_amp', action = 'store_true', help = 'use automatic mixed precision training', default = False)
    parser.add_argument('--use_gpu', type = bool, default = True, help = 'use gpu')
    parser.add_argument('--gpu', type = int, default = 0, help = 'gpu')
    parser.add_argument('--devices', type = str, default = '0,1', help = 'device ids of multile gpus')
    parser.add_argument('--embed', type = str, default = 0)
    parser.add_argument('--loss', type = str, default = 'smoothL1', choices = ['mse', 'smoothL1'])
    parser.add_argument('--pct_start', type = float, default = 0.2, help = 'pct_start')
    
    ''' Optuna Hyperparameters: if you don't pass the argument, then value form the hyperparameters_optuna.py will be considered as search region'''
    parser.add_argument('--optuna_seq_len', type = int, nargs = '+', required = False, default = None, help = 'Optuna seq length list')
    parser.add_argument('--optuna_lr', type = float, nargs = '+', required = False, default = None, help = 'Optuna lr: first-min, 2nd-max')
    parser.add_argument('--optuna_batch', type = int, nargs = '+', required = False, default = None, help = 'Optuna batch size list')
    parser.add_argument('--optuna_wavelet', type = str, nargs = '+', required = False, default = None, help = 'Optuna wavelet type list')
    parser.add_argument('--optuna_tfactor', type = int, nargs = '+', required = False, default = None, help = 'Optuna tfactor list')
    parser.add_argument('--optuna_dfactor', type = int, nargs = '+', required = False, default = None, help = 'Optuna dfactor list')
    parser.add_argument('--optuna_epochs', type = int, nargs = '+', required = False, default = None, help = 'Optuna epochs list')
    parser.add_argument('--optuna_dropout', type = float, nargs = '+', required = False, default = None, help = 'Optuna dropout list')
    parser.add_argument('--optuna_embedding_dropout', type = float, nargs = '+', required = False, default = None, help = 'Optuna embedding_dropout list')
    parser.add_argument('--optuna_patch_len', type = int, nargs = '+', required = False, default = None, help = 'Optuna patch len list')
    parser.add_argument('--optuna_stride', type = int, nargs = '+', required = False, default = None, help = 'Optuna stride len list')
    parser.add_argument('--optuna_lradj', type = str, nargs = '+', required = False, default = None, help = 'Optuna lr adjustment type list')
    parser.add_argument('--optuna_dmodel', type = int, nargs = '+', required = False, default = None, help = 'Optuna dmodel list')
    parser.add_argument('--optuna_weight_decay', type = float, nargs = '+', required = False, default = None, help = 'Optuna weight_decay list')
    parser.add_argument('--optuna_patience', type = int, nargs = '+', required = False, default = None, help = 'Optuna patience list')
    parser.add_argument('--optuna_level', type = int, nargs = '+', required = False, default = None, help = 'Optuna level list')    
    parser.add_argument('--optuna_trial_num', type = int, required = False, default = None, help = 'Optuna trial number')        
    args = parser.parse_args()
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ','')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    data_parser = {
        'ETTh1': {'data': 'ETTh1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTh2': {'data': 'ETTh2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm1': {'data': 'ETTm1.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'ETTm2': {'data': 'ETTm2.csv', 'root_path': './data/ETT/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Weather': {'data': 'weather.csv', 'root_path': './data/weather/', 'T': 'OT', 'M': [21, 21], 'S': [1, 1], 'MS': [21, 1]},
        'Traffic': {'data': 'traffic.csv', 'root_path': './data/traffic/', 'T': 'OT', 'M': [862, 862], 'S': [1, 1], 'MS': [862, 1]},
        'Electricity': {'data': 'electricity.csv', 'root_path': './data/electricity/', 'T': 'OT', 'M': [321, 321], 'S': [1, 1], 'MS': [321, 1]},
        'ILI':  {'data': 'national_illness.csv', 'root_path': './data/illness/', 'T': 'OT', 'M': [7, 7], 'S': [1, 1], 'MS': [7, 1]},
        'Solar':  {'data': 'solar_AL.txt', 'root_path': './data/solar/', 'T': None, 'M': [137, 137], 'S': [None, None], 'MS': [None, None]},
    }
    
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info['data']
        args.root_path = data_info['root_path']
        args.target = data_info['T']
        args.c_in = data_info[args.features][0]
        args.c_out = data_info[args.features][1]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]
    
    
    if args.use_hyperParam_optim == False: 
        print('Args in experiment: {}'.format(args))
        setting = '{}_{}_dec-{}_sl{}_pl{}_dm{}_bt{}_wv{}_tf{}_df{}_ptl{}_stl{}_sd{}'.format(args.model, args.data, not args.no_decomposition, args.seq_len, args.pred_len, args.d_model, args.batch_size, args.wavelet, args.tfactor, args.dfactor, args.patch_len, args.stride, args.seed)
        
        set_random_seed(args.seed)
        Exp = Exp_Main
        exp = Exp(args) 
        
        print('Start Training- {}'.format(setting))
        exp.train(setting)
        
        print('Start Testing- {}'.format(setting))
        loss_mse, loss_mae = exp.test(setting) # mse
    
        
    elif args.use_hyperParam_optim:
        ''' Tuning the model using Optuna hyperparameter tuning framework'''
        tuner = Tuner(42, args.n_jobs)
        tuner.tune(args)
        