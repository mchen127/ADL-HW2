#!/bin/bash

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found, installing it now..."
    pip install gdown
fi

# Download the file from Google Drive
echo "Downloading public.jsonl from Google Drive..."
gdown --id 1FhXpwJbF1ewHkvBHeMMb4ByVtcaeJy6Y -O data/public.jsonl

echo "Download completed: public.jsonl"