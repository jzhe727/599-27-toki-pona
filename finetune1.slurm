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

export BS=128
export GAS=1
export m=Helsinki-NLP/opus-mt-es-en
export MAX_LEN=128
export OUT_DIR=models_finetune_small_opus-mt-es-en
export TOK_EN_DIR=hf_tok_en

python finetune_trainer.py \
    --tokenizer_name $m --model_name_or_path $m \
    --data_dir $TOK_EN_DIR \
    --output_dir OUT_DIR --overwrite_output_dir \
    --learning_rate=3e-4 \
    --warmup_steps 500 --sortish_sampler \
    --fp16 \
    --gradient_accumulation_steps=$GAS \
    --per_device_train_batch_size=$BS --per_device_eval_batch_size=$BS \
    --freeze_encoder --freeze_embeds \
    --num_train_epochs=10 \
    --save_strategy epoch --eval_strategy epoch \
    --save_steps 1 --eval_steps 1 \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN \
    --val_max_target_length $MAX_TGT_LEN --test_max_target_length $MAX_TGT_LEN \
    --do_train --do_eval --do_predict \
    --evaluation_strategy steps \
    --predict_with_generate --logging_first_step \
    --task translation --label_smoothing_factor 0.1 \
    "$@"


##### /scratch/${SLURM_JOB_ID}
##### 5 seconds per epoch with batch size 64