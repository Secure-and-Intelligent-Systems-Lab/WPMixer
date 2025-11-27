<h1 align="center">WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting</h1>



This document explains how to run the hyper-tuning scripts for WPMixer, how to organize logs, how to specify datasets, and how each parameter works.
# 1. Overview
WPMixer supports automatic hyper-parameter optimization using Optuna.
You can run hyper-tuning for one or multiple datasets and automatically search over:
- Learning rates
- Sequence lengths
- Batch size
- Wavelet types
- Decomposition levels
- Patch lengths & strides
- Dropout choices
- Temporal expansion and embedding expansion factors

All results are logged under ./logs/WPMixer/.

# 2. Example of the Hyper-parameter Tuning Script:
Following script will also be found in ```./scripts/HyperParameter_Tuning/ETT_optuna_unified.sh```

```bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/WPMixer" ]; then
    mkdir ./logs/WPMixer
fi
export CUDA_VISIBLE_DEVICES=0

# General
model_name=WPMixer
task_name=long_term_forecast

python -u main_run2.py \
	--task_name $task_name \
	--model $model_name \
	--use_hyperParam_optim \
	--datasets ETTh1 ETTh2 \
	--pred_lens 192 \
	--loss smoothL1 \
	--use_amp \
	--n_jobs 1 \
	--optuna_lr 0.00001 0.01 \
	--optuna_batch 128 \
	--optuna_wavelet db2 db3 db5 sym2 sym3 sym4 sym5 coif4 coif5 \
	--optuna_seq_len 96 192 336 \
	--optuna_tfactor 3 5 7 \
	--optuna_dfactor 3 5 7 8 \
	--optuna_epochs 10 \
	--optuna_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_embedding_dropout 0.0 0.05 0.1 0.2 0.4 \
	--optuna_patch_len 16 \
	--optuna_stride 8 \
	--optuna_lradj type3 \
	--optuna_dmodel 128 256 \
	--optuna_weight_decay 0.0 \
	--optuna_patience 5 \
	--optuna_level 1 2 3 \
	--optuna_trial_num 200 >logs/WPMixer/ETTh_192_with_decomposition.log
```

# 3. Parameter Descriptions
This section explains every parameter used in the hyper-tuning script.

## General Parameters
| Parameter     | Description                                   |
| ------------- | --------------------------------------------- |
| `--task_name` | Task type (e.g., long_term_forecast).         |
| `--model`     | Model name (WPMixer).                         |
| `--loss`      | Loss function (smoothL1, MSE).                |
| `--use_amp`   | Enables automatic mixed precision for speed.  |
| `--datasets`  | One or multiple datasets to tune together.    |
| `--pred_lens` | Prediction horizon (e.g., 96).                |
| `--n_jobs`    | Number of parallel Optuna workers.            |

## Optuna Search Space Parameters
| Parameter                    | Description                                          |
| ---------------------------- | ---------------------------------------------------- |
| `--optuna_lr lr_min lr_max`  | Learning rate search interval from minimum LR to max.LR.                       |
| `--optuna_batch`             | Candidate batch sizes.                               |
| `--optuna_wavelet`           | Wavelet families for multi-resolution decomposition. Set a list of wavelet type to find the optimum one. |
| `--optuna_seq_len`           | Multiple input sequence lengths to find the optimum one.                     |
| `--optuna_tfactor`           | Multiple temporal mixing factors to find the optimum one.                              |
| `--optuna_dfactor`           | Multiple embedding mixing factors to find the optimum one.                         |
| `--optuna_epochs`            | Max epochs for each trial.                           |
| `--optuna_dropout`           | Multiple Dropout search values to find the optimum one.                               |
| `--optuna_embedding_dropout` | Multiple Embedding Dropout search values to find the optimum one.                     |
| `--optuna_patch_len`         | Length of each patch.              |
| `--optuna_stride`            | Stride between patches.                              |
| `--optuna_lradj`             | Learning rate scheduler policy.                      |
| `--optuna_dmodel`            | Multiple embedding dimensions to find the optimum one.                    |
| `--optuna_weight_decay`      | Weight decay coefficient for AdamW.                  |
| `--optuna_patience`          | Early stopping patience.                             |
| `--optuna_level`             | Multiple Wavelet Decomposition levels to find the optimum one.        |
| `--optuna_trial_num`         | Total number of Optuna trials.                       |

## ⚠️ Note (Important for WPMixer Hyper-Tuning)
When selecting ```optuna_patch_len``` and ```optuna_level```, ensure their compatibility with the effective sequence length at the deepest wavelet decomposition branch.
If you use $m$ levels in:
```
--optuna_level m
```
Then:
* The model uses $m$ detailed resolution branches
* And $1$ approximation branch
* The minimum effective sequence length among all detailed branches becomes: ```optuna_seq_len / 2^m```

To avoid invalid patching, you must satisfy: **(optuna_seq_len / 2^m) > optuna_patch_len**

Example:
```
optuna_seq_len = 96
optuna_level = 3   →   effective_seq_len = 96 / 8 = 12
optuna_patch_len = 16   ❌ invalid (12 < 16)
```
