#!/bin/bash

# Define the file ID and destination filename
FILE_ID="1t8QSuHXz7L9nYRrAwLQ4ponSweAp2WW_"
DESTINATION="data.zip"

# Download the file from Google Drive using gdown
echo "Downloading data.zip from Google Drive..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O ${DESTINATION}

# Check if the download was successful
if [ -f "${DESTINATION}" ]; then
    echo "Download completed. Unzipping ${DESTINATION}..."
    unzip ${DESTINATION} -d ./
    # Remove the __MACOSX folder if it exists
    if [ -d "__MACOSX" ]; then
    echo "Removing __MACOSX folder..."
    rm -rf __MACOSX
    fi
    echo "Unzipping completed. Files are in the ./ directory."
    echo "Deleting data.zip..."
    rm data.zip
else
    echo "Download failed. Please check the file ID or your network connection."
    exit 1
fi
