#!/bin/bash

# Create logs directory if it doesn't exist
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

# Create WPMixer logs directory if it doesn't exist
if [ ! -d "./logs/WPMixer" ]; then
    mkdir ./logs/WPMixer
fi

# Set the GPU to use
export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=WPMixer

# Datasets and prediction lengths
dataset=ETTm1
seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.000358938 0.002611066 0.000191965 0.009967788)
batches=(128 128 128 128)
wavelets=(db2 db2 db2 db3)
levels=(1 3 2 2)
tfactors=(3 3 7 3)
dfactors=(3 3 7 3)
epochs=(80 80 80 80)
dropouts=(0.4 0.05 0.0 0.4)
embedding_dropouts=(0.2 0.1 0.4 0.2)
patch_lens=(48 48 48 48)
strides=(24 24 24 24)
lradjs=(type3 type3 type3 type3)
d_models=(128 128 128 128)
patiences=(12 12 12 12)
weight_decays=(0.0001 0.0001 0.0001 0.0001)

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	log_file="logs/${model_name}/univariate_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--features S \
		--d_model ${d_models[$i]} \
		--tfactor ${tfactors[$i]} \
		--dfactor ${dfactors[$i]} \
		--wavelet ${wavelets[$i]} \
		--level ${levels[$i]} \
		--patch_len ${patch_lens[$i]} \
		--stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--embedding_dropout ${embedding_dropouts[$i]} \
		--weight_decay ${weight_decays[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done
