#!/bin/bash

# Ensure required utilities are installed
if ! command -v gdown &> /dev/null; then
    echo "gdown could not be found. Please install gdown with: pip install gdown"
    exit 1
fi

# Download the model from Google Drive using gdown
echo "Downloading model..."
gdown --id 18WUFm6gOh6Q0T9SF0KhC8B5kUo-rEhXw -O fine_tuned_mt5_v3_epoch_6.pth

# Download the zipped tokenizer from Google Drive
echo "Downloading tokenizer..."
gdown --id 114GsU_yEgWYl65l4xwNkfjcUcQ2kdtD0 -O tokenizer.zip

# Unzip the tokenizer
echo "Unzipping tokenizer..."
unzip tokenizer.zip -d ./

# Clean up the zip file
rm tokenizer.zip

echo "Downloads complete!"
