#!/bin/bash

# Usage: ./train.sh

MODEL_NAME="google/mt5-small"
DATA_PATH="./data/train.jsonl"
SAVE_PATH="fine_tuned_mt5"
BATCH_SIZE=8
LEARNING_RATE=5e-5
EPOCHS=3
MAX_INPUT_LENGTH=256
MAX_OUTPUT_LENGTH=64
GRADIENT_ACCUMULATION_STEPS=5
VALIDATION_SPLIT=0.1
USE_FP16="--use_fp16"
USE_ADAFACTOR="--use_adafactor"

python3 train.py \
  --model_name $MODEL_NAME \
  --data_path $DATA_PATH \
  --save_path $SAVE_PATH \
  --batch_size $BATCH_SIZE \
  --learning_rate $LEARNING_RATE \
  --epochs $EPOCHS \
  --max_input_length $MAX_INPUT_LENGTH \
  --max_output_length $MAX_OUTPUT_LENGTH \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --validation_split $VALIDATION_SPLIT \
  $USE_FP16 \
  $USE_ADAFACTOR