#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run.sh /path/to/input.jsonl /path/to/output.jsonl"
    exit 1
fi

# Assign arguments to variables
INPUT_PATH=$1
OUTPUT_PATH=$2

# Run the Python script with the appropriate arguments
python inference.py \
    --eval_dataset_path "$INPUT_PATH" \
    --submission_path "$OUTPUT_PATH" \
    --model_name "google/mt5-small" \
    --model_path "./fine_tuned_mt5_v3_epoch_6.pth" \
    --tokenizer_path "./fine_tuned_mt5_v3_epoch_6" \
    --batch_size 8 \
    --max_input_length 512 \
    --max_output_length 100 \
    --num_beams 20 \
    --early_stopping \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50
