#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu-v100

source ~/software/miniconda3/bin/activate
cd ~/599-27-toki-pona
conda activate nlpa4

export BS=64
export GAS=1
export m=Helsinki-NLP/opus-mt-zh-en
export MAX_LEN=128
export MAX_TGT_LEN=128 
export OUT_DIR=models_zh-en
export TOK_EN_DIR=${PWD}/hf_tok_en

python transformers/examples/legacy/seq2seq/finetune_trainer.py \
    --tokenizer_name $m --model_name_or_path $m \
    --data_dir $TOK_EN_DIR \
    --output_dir $OUT_DIR --overwrite_output_dir \
    --learning_rate=3e-4 \
    --warmup_steps 500 --sortish_sampler \
    --fp16 \
    --gradient_accumulation_steps=$GAS \
    --per_device_train_batch_size=$BS --per_device_eval_batch_size=$BS \
    --freeze_encoder --freeze_embeds \
    --num_train_epochs=40 \
    --save_strategy epoch --evaluation_strategy epoch \
    --save_steps 1 --eval_steps 1 \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN \
    --val_max_target_length $MAX_TGT_LEN --test_max_target_length $MAX_TGT_LEN \
    --do_train --do_eval --do_predict \
    --predict_with_generate --logging_first_step \
    --task translation --label_smoothing_factor 0.1 \
    "$@"


##### /scratch/${SLURM_JOB_ID}
##### 5 seconds per epoch with batch size 64