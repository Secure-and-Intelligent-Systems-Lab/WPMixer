

import os
import torch
import argparse
import optuna

torch.set_printoptions(precision = 10)
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast

from utils.Tuner import TunerSTF # additional line for optuna
from utils.tools import set_random_seed


if __name__ == "__main__":
    seed = 42
    set_random_seed(seed)
    
    parser = argparse.ArgumentParser(description='[WPMixer] Long Sequences Forecasting')
    # frequent changing hy.params
    parser.add_argument('--task_name', type=str, required=False, choices=['short_term_forecast'], default='short_term_forecast')
    parser.add_argument('--model', type=str, required=False, choices=['WPMixer'], default='WPMixer',help='model of experiment')
    parser.add_argument('--use_hyperParam_optim', action = 'store_true', default = False, help = 'True: HyperParameters optimization using optuna, False: no optimization')
    parser.add_argument('--data', type=str, required=False, choices=['m4'], default='m4',help='m4')
    
    parser.add_argument('--uncertainty', type = int, required = False, choices = [0, 1], default = 1, help = '0: non distributional uncertainty model, 1: without uncertainty model')
    parser.add_argument('--no_decomposition', action = 'store_true', default = False, help = 'whether to use wavelet decomposition')
    parser.add_argument('--use_multi_seeds', action = 'store_true', default = False, help = 'whether to use multiple random seeds')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--n_jobs', type = int, required = False, default = 1, help = 'number_of_jobs for optuna')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--wavelet', type=str, default='db3', help = 'type of discrete wavelet')
    parser.add_argument('--level', type=int, default=1, help='level for multilevel wavelet decomposition')
    parser.add_argument('--tfactor', type=int, default=5, help='t factor in mixer')
    parser.add_argument('--dfactor', type=int, default=5, help='d factor in mixer')
    parser.add_argument('--train_epochs', type=int, default=50, help='number of train epochs')
    parser.add_argument('--dropout', type=float, default=0.02, help='dropout of the mixer')
    parser.add_argument('--embedding_dropout', type=float, default=0.02, help='dropout of the embedding')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='patch stride')
    parser.add_argument('--d_model', type=int, default=32, help='dmodel')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay for L2 regularization')
    parser.add_argument('--patience', type=int, default=20, help='patience of the model training')
    
    # rare changing hy.params
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--cols', type=str, nargs='+', default = None, help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--lradj', type=str, default='type3',help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--beta1', type=float, default=0.9, help='adam optimizer betas')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam optimizer betas')
    
    # parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1',help='device ids of multile gpus')
    parser.add_argument('--embed', type=str, default=0)
    parser.add_argument('--loss', type=str, default='SMAPE', choices=['mse', 'smoothL1', 'SMAPE']) # SMAPE: for m4 forecasting, other: smoothL1
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    
    # non distributional uncertainty
    parser.add_argument('--Pc', type=float, default=0.9, help='Desired captured (within prediction interval) data portion')
    parser.add_argument('--_lambda', type=float, default=1e-5, help='Factor for PICP')
    parser.add_argument('--s', type=float, default=1, help='Factor')
    parser.add_argument('--alpha', type=float, default=1, help='Factor MPIWc')
    parser.add_argument('--alpha_ext', type=float, default=10, help='Factor MPIWc_ext')
    parser.add_argument('--beta', type=float, default=1, help='Factor for LPE (Loss point estimate)')
    parser.add_argument('--soft', type=bool, default=True, help='Soft method or hard method')
    
    # bayesian uncertainty
    parser.add_argument('--alpha_bayesian', type=float, default=1, help='Factor for Bayesian model')
    parser.add_argument('--beta_bayesian', type=float, default=1, help='Factor for Bayesian model')
    
    # Optuna Search Region: 
    ''' if you don't pass the argument, then value form the hyperparameters_optuna.py will be considered as search region'''
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
    parser.add_argument('--optuna_useVaryingBetas', action = 'store_true', default = False, help = 'varying beta for adam optimizer')
    #
    args = parser.parse_args()
    ##############################################################################################
    ##############################################################################################
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
        'm4': {'data': 'm4', 'root_path': './data/m4/', 'T': 'Null', 'M': [1, 1], 'S': [1, 1], 'MS': [1, 1]},
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
    
    if args.use_hyperParam_optim == False: # this block is not for hyper param tuning
        print('Args in experiment:')
        print(args)
        
        setting = '{}_{}_dec-{}_{}_dm{}_bt{}_wv{}_tf{}_df{}_ptl{}_stl{}_sd{}'.format(args.model, args.data, not args.no_decomposition, args.seasonal_patterns, args.d_model, args.batch_size, args.wavelet, args.tfactor, args.dfactor, args.patch_len, args.stride, seed) 
        
        Exp = Exp_Short_Term_Forecast
        exp = Exp(args) # set experiments
        
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting) # mse
        torch.cuda.empty_cache() 
        
    elif args.use_hyperParam_optim: # this is for tuner only
        tuner = TunerSTF(42, args.n_jobs) if args.use_hyperParam_optim else None # declaring the Tuner only if it is required
        tuner.tune(args)
        