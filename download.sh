#!/bin/bash

# Ensure required utilities are installed
if ! command -v gdown &> /dev/null; then
    echo "gdown could not be found. Please install gdown with: pip install gdown"
    exit 1
fi

if ! command -v wget &> /dev/null; then
    echo "wget could not be found. Please install wget with: sudo apt-get install wget"
    exit 1
fi

# Download the fine-tuned model
echo "Downloading fine-tuned model..."
gdown --id 18WUFm6gOh6Q0T9SF0KhC8B5kUo-rEhXw -O fine_tuned_mt5_v3_epoch_6.pth

# Download the tokenizer
echo "Downloading tokenizer..."
gdown --id 114GsU_yEgWYl65l4xwNkfjcUcQ2kdtD0 -O tokenizer.zip

# Unzip the tokenizer
echo "Unzipping tokenizer..."
unzip tokenizer.zip -d ./

# Clean up
rm tokenizer.zip

# Download base model files for google/mt5-small
echo "Downloading base MT5 model (google/mt5-small)..."
MODEL_DIR="google_mt5_small"
mkdir -p $MODEL_DIR
wget -P $MODEL_DIR https://huggingface.co/google/mt5-small/resolve/main/config.json
wget -P $MODEL_DIR https://huggingface.co/google/mt5-small/resolve/main/pytorch_model.bin
wget -P $MODEL_DIR https://huggingface.co/google/mt5-small/resolve/main/tokenizer_config.json
wget -P $MODEL_DIR https://huggingface.co/google/mt5-small/resolve/main/spiece.model
wget -P $MODEL_DIR https://huggingface.co/google/mt5-small/resolve/main/special_tokens_map.json


echo "Downloads complete!"
