#!/bin/bash

# Ensure gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown is not installed. Install it using 'pip install gdown' and try again."
    exit 1
fi

# Ensure wget is installed
if ! command -v wget &> /dev/null
then
    echo "wget is not installed. Install it using 'pip install wget' and try again."
    exit 1
fi

# Download the tokenizer folder
echo "Downloading tokenizer folder..."
gdown --folder --id 1-fsq-QPgFSK603ynbnW6afoX6zeTf8n- || {
    echo "Failed to download the tokenizer folder."
    exit 1
}

# Download the fine-tuned model file
echo "Downloading fine-tuned model..."
gdown --id 1-azNhu6YxVk93YwQi5jxD-OxAqk_1SpL || {
    echo "Failed to download the fine-tuned model."
    exit 1
}

echo "Download complete."


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
