
import torch
from utils.tools import set_random_seed
import optuna
from collections import defaultdict
import pandas as pd
torch.set_printoptions(precision = 10)
from exp.exp_main import Exp_Main
import datetime


class Tuner:
    # Tuner for Long Term forecasting
    def __init__(self, ranSeed, n_jobs):
        self.fixedSeed = ranSeed
        self.n_jobs = n_jobs 
        self.result_dic = defaultdict(list)
        self.current_time = datetime.datetime.now()
        self.current_time = str(self.current_time).replace(':', '-').split('.')[0]
        
    def optuna_objective(self, trial, args):
        # these are the params that will be tuned:
        args.seq_len = trial.suggest_categorical('seq_len', args.optuna_seq_len)
        args.learning_rate = trial.suggest_loguniform('lr', args.optuna_lr[0], args.optuna_lr[1]) 
        args.batch_size = trial.suggest_categorical('batch', args.optuna_batch)
        args.tfactor = trial.suggest_categorical('tfactor', args.optuna_tfactor) 
        args.dfactor = trial.suggest_categorical('dfactor', args.optuna_dfactor) 
        args.train_epochs = trial.suggest_categorical('epochs', args.optuna_epochs) 
        args.dropout = trial.suggest_categorical('dropout', args.optuna_dropout) 
        args.embedding_dropout = trial.suggest_categorical('embedding_dropout', args.optuna_embedding_dropout) 
        args.patch_len = trial.suggest_categorical('patch_len', args.optuna_patch_len) 
        args.stride = trial.suggest_categorical('stride', args.optuna_stride)
        args.lradj = trial.suggest_categorical('lradj', args.optuna_lradj) 
        args.d_model = trial.suggest_categorical('d_model', args.optuna_dmodel) 
        args.weight_decay = trial.suggest_categorical('weight_decay', args.optuna_weight_decay) 
        args.patience = trial.suggest_categorical('patience', args.optuna_patience) 
        if args.no_decomposition: # no wavelet decomposition
            args.wavelet = 'db2' # this doesn't have any impact
            args.level = 1 # doesn't have any impact
        else:
            args.wavelet = trial.suggest_categorical('wavelet', args.optuna_wavelet) 
            args.level = trial.suggest_categorical('level', args.optuna_level) 
        # params end
        
        setting = '{}_{}_sl{}_pl{}_dm{}_bt{}_wv{}_tf{}_df{}_ptl{}_stl{}_sd{}'.format(args.model, args.data, args.seq_len, args.pred_len, args.d_model, args.batch_size, args.wavelet, args.tfactor, args.dfactor, args.patch_len, args.stride, self.fixedSeed)
        
        set_random_seed(self.fixedSeed) # 42
        Exp = Exp_Main
        exp = Exp(args) # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting, optunaTrialReport = trial)        
        
        return exp.min_test_loss
    
    
    def tune(self, args): # args with some fixed params. other will be tuned in objective function
        n = args.optuna_trial_num
        try:
            del self.study
            print('deleted previous tuner obj')
        except:
            print('no prev tuner obj')
        self.study = optuna.create_study(direction='minimize', sampler = optuna.samplers.TPESampler(seed = 42))
        wrapped_objective = lambda trial: self.optuna_objective(trial, args)
        self.study.optimize(wrapped_objective, n_trials=n, n_jobs = self.n_jobs) 
        self.save_result(args)
        return 
    
    def save_result(self, args):
        file_name = '{}_{}_s{}_decom-{}'.format(args.model, args.data, args.seq_len, not args.no_decomposition)
        data, pred_len = args.data, args.pred_len
        # best trial
        best_trial = self.study.best_trial
        # best_params
        best_params = self.study.best_params
        # best result 
        best_result = self.study.best_trial.value
        # saving in dictionary
        self.result_dic['data'].append(data)
        self.result_dic['pred_len'].append(pred_len)
        self.result_dic['loss'].append(best_result)
        
        for key, value in best_params.items():
            self.result_dic[key].append(value)
            
        result_df = pd.DataFrame(self.result_dic)
        try:
            result_df.to_csv('./hyperParameterSearchOutput/' + file_name + '_bst_parms_'+ self.current_time + '.csv')
        except:
            print('save failed: close best param csv file')
        print(result_df)
            