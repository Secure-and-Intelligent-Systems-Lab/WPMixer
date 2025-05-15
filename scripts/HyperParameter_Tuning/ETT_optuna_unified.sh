if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/WLMixer" ]; then
    mkdir ./logs/WLMixer
fi
export CUDA_VISIBLE_DEVICES=0
# General
model_name=WLMixer
task_name=long_term_forecast
loss_name=smoothL1
patience=5
lradj=type3
n_jobs=1


# unified-TimeMixer setting
learning_rate1=0.00001
learning_rate2=0.01
batch_size=128
epochs=10

# specific- ETTh1 and ETTh2
patch_len=16
stride=8
trial_num=200
seq_len=96


# 96
python -u main_run2.py \
	--task_name $task_name \
	--model $model_name \
	--use_hyperParam_optim \
	--datasets ETTh1 ETTh2 \
	--pred_lens 96 \
	--loss $loss_name \
	--use_amp \
	--n_jobs $n_jobs \
	--optuna_lr $learning_rate1 $learning_rate2 \
	--optuna_batch $batch_size \
	--optuna_wavelet db2 db3 db5 sym2 sym3 sym4 sym5 coif4 coif5 \
	--optuna_seq_len $seq_len \
	--optuna_tfactor 3 5 7 \
	--optuna_dfactor 3 5 7 8 \
	--optuna_epochs $epochs \
	--optuna_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_embedding_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_patch_len $patch_len \
	--optuna_stride $stride \
	--optuna_lradj $lradj \
	--optuna_dmodel 128 256 \
	--optuna_weight_decay 0.0 \
	--optuna_patience $patience \
	--optuna_level 1 \
	--optuna_trial_num $trial_num >logs/WLMixer/ETTh_96_with_decomposition.log

#192
python -u main_run2.py \
	--task_name $task_name \
	--model $model_name \
	--use_hyperParam_optim \
	--datasets ETTh1 ETTh2 \
	--pred_lens 192 \
	--loss $loss_name \
	--use_amp \
	--n_jobs $n_jobs \
	--optuna_lr $learning_rate1 $learning_rate2 \
	--optuna_batch $batch_size \
	--optuna_wavelet db2 db3 db5 sym2 sym3 sym4 sym5 coif4 coif5 \
	--optuna_seq_len $seq_len \
	--optuna_tfactor 3 5 7 \
	--optuna_dfactor 3 5 7 8 \
	--optuna_epochs $epochs \
	--optuna_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_embedding_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_patch_len $patch_len \
	--optuna_stride $stride \
	--optuna_lradj $lradj \
	--optuna_dmodel 128 256 \
	--optuna_weight_decay 0.0 \
	--optuna_patience $patience \
	--optuna_level 1 2 3 \
	--optuna_trial_num $trial_num >logs/WLMixer/ETTh_192_with_decomposition.log

#336
python -u main_run2.py \
	--task_name $task_name \
	--model $model_name \
	--use_hyperParam_optim \
	--datasets ETTh1 ETTh2 \
	--pred_lens 336 \
	--loss $loss_name \
	--use_amp \
	--n_jobs $n_jobs \
	--optuna_lr $learning_rate1 $learning_rate2 \
	--optuna_batch $batch_size \
	--optuna_wavelet db2 db3 db5 sym2 sym3 sym4 sym5 coif4 coif5 \
	--optuna_seq_len $seq_len \
	--optuna_tfactor 3 5 7 \
	--optuna_dfactor 3 5 7 8 \
	--optuna_epochs $epochs \
	--optuna_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_embedding_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_patch_len $patch_len \
	--optuna_stride $stride \
	--optuna_lradj $lradj \
	--optuna_dmodel 128 256 \
	--optuna_weight_decay 0.0 \
	--optuna_patience $patience \
	--optuna_level 1 2 3 \
	--optuna_trial_num $trial_num >logs/WLMixer/ETTh_336_with_decomposition.log

# 720
python -u main_run2.py \
	--task_name $task_name \
	--model $model_name \
	--use_hyperParam_optim \
	--datasets ETTh1 ETTh2 \
	--pred_lens 720 \
	--loss $loss_name \
	--use_amp \
	--n_jobs $n_jobs \
	--optuna_lr $learning_rate1 $learning_rate2 \
	--optuna_batch $batch_size \
	--optuna_wavelet db2 db3 db5 sym2 sym3 sym4 sym5 coif4 coif5 \
	--optuna_seq_len $seq_len \
	--optuna_tfactor 3 5 7 \
	--optuna_dfactor 3 5 7 8 \
	--optuna_epochs $epochs \
	--optuna_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_embedding_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_patch_len $patch_len \
	--optuna_stride $stride \
	--optuna_lradj $lradj \
	--optuna_dmodel 128 256 \
	--optuna_weight_decay 0.0 \
	--optuna_patience $patience \
	--optuna_level 1 2 3 \
	--optuna_trial_num $trial_num >logs/WLMixer/ETTh_720_with_decomposition.log
	
#######################################
#######################################
	
	
# specific- ETTm1 and ETTm2
patch_len=48
stride=24
trial_num=150
seq_len=96
# 96
python -u main_run2.py \
	--task_name $task_name \
	--model $model_name \
	--use_hyperParam_optim \
	--datasets ETTm1 ETTm2 \
	--pred_lens 96 \
	--loss $loss_name \
	--use_amp \
	--n_jobs $n_jobs \
	--optuna_lr $learning_rate1 $learning_rate2 \
	--optuna_batch $batch_size \
	--optuna_wavelet db2 db3 db5 sym2 sym3 sym4 sym5 coif4 coif5 \
	--optuna_seq_len $seq_len \
	--optuna_tfactor 3 5 7 \
	--optuna_dfactor 3 5 7 8 \
	--optuna_epochs $epochs \
	--optuna_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_embedding_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_patch_len $patch_len \
	--optuna_stride $stride \
	--optuna_lradj $lradj \
	--optuna_dmodel 128 256 \
	--optuna_weight_decay 0.0 \
	--optuna_patience $patience \
	--optuna_level 1 \
	--optuna_trial_num $trial_num >logs/WLMixer/ETTm_96_with_decomposition.log

# 192
python -u main_run2.py \
	--task_name $task_name \
	--model $model_name \
	--use_hyperParam_optim \
	--datasets ETTm1 ETTm2 \
	--pred_lens 192 \
	--loss $loss_name \
	--use_amp \
	--n_jobs $n_jobs \
	--optuna_lr $learning_rate1 $learning_rate2 \
	--optuna_batch $batch_size \
	--optuna_wavelet db2 db3 db5 sym2 sym3 sym4 sym5 coif4 coif5 \
	--optuna_seq_len $seq_len \
	--optuna_tfactor 3 5 7 \
	--optuna_dfactor 3 5 7 8 \
	--optuna_epochs $epochs \
	--optuna_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_embedding_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_patch_len $patch_len \
	--optuna_stride $stride \
	--optuna_lradj $lradj \
	--optuna_dmodel 128 256 \
	--optuna_weight_decay 0.0 \
	--optuna_patience $patience \
	--optuna_level 1 2 \
	--optuna_trial_num $trial_num >logs/WLMixer/ETTm_192_with_decomposition.log
	
# 336
python -u main_run2.py \
	--task_name $task_name \
	--model $model_name \
	--use_hyperParam_optim \
	--datasets ETTm1 ETTm2 \
	--pred_lens 336 \
	--loss $loss_name \
	--use_amp \
	--n_jobs $n_jobs \
	--optuna_lr $learning_rate1 $learning_rate2 \
	--optuna_batch $batch_size \
	--optuna_wavelet db2 db3 db5 sym2 sym3 sym4 sym5 coif4 coif5 \
	--optuna_seq_len $seq_len \
	--optuna_tfactor 3 5 7 \
	--optuna_dfactor 3 5 7 8 \
	--optuna_epochs $epochs \
	--optuna_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_embedding_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_patch_len $patch_len \
	--optuna_stride $stride \
	--optuna_lradj $lradj \
	--optuna_dmodel 128 256 \
	--optuna_weight_decay 0.0 \
	--optuna_patience $patience \
	--optuna_level 1 2 \
	--optuna_trial_num $trial_num >logs/WLMixer/ETTm_336_with_decomposition.log
	
# 720
python -u main_run2.py \
	--task_name $task_name \
	--model $model_name \
	--use_hyperParam_optim \
	--datasets ETTm1 ETTm2 \
	--pred_lens 720 \
	--loss $loss_name \
	--use_amp \
	--n_jobs $n_jobs \
	--optuna_lr $learning_rate1 $learning_rate2 \
	--optuna_batch $batch_size \
	--optuna_wavelet db2 db3 db5 sym2 sym3 sym4 sym5 coif4 coif5 \
	--optuna_seq_len $seq_len \
	--optuna_tfactor 3 5 7 \
	--optuna_dfactor 3 5 7 8 \
	--optuna_epochs $epochs \
	--optuna_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_embedding_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_patch_len $patch_len \
	--optuna_stride $stride \
	--optuna_lradj $lradj \
	--optuna_dmodel 128 256 \
	--optuna_weight_decay 0.0 \
	--optuna_patience $patience \
	--optuna_level 1 2 \
	--optuna_trial_num $trial_num >logs/WLMixer/ETTm_720_with_decomposition.log