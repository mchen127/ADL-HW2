#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: bash ./run.sh <input_file> <output_file>"
    exit 1
fi

# Assign arguments to variables
INPUT_FILE=$1
OUTPUT_FILE=$2

# Run the inference script with the provided input and output paths
python3 inference.py --input $INPUT_FILE --output $OUTPUT_FILE