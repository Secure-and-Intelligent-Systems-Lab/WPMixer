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
dataset=ETTh2
seq_lens=(96 96 96 96)
pred_lens=(96 192 336 720)
learning_rates=(0.000478097 0.001470515 0.001194632 0.000296491)
batches=(128 128 128 128)
wavelets=(db3 coif4 coif4 coif5)
levels=(1 2 2 1)
tfactors=(5 3 5 7)
dfactors=(8 8 3 5)
epochs=(10 10 10 10)
dropouts=(0.4 0.1 0.1 0.4)
embedding_dropouts=(0.0 0.05 0.0 0.0)
patch_lens=(16 16 16 16)
strides=(8 8 8 8)
lradjs=(type3 type3 type3 type3)
d_models=(256 256 256 256)
patiences=(5 5 5 5)


# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
	log_file="logs/${model_name}/unify_result_${dataset}_${pred_lens[$i]}.log"
	python -u run_LTF.py \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
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
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--use_amp > $log_file
done
