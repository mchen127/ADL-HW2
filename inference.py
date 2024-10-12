import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import argparse

def load_model(model_path="./model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    return tokenizer, model

def read_input_file(input_file_path):
    with open(input_file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def write_output_file(output_data, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def inference(model, tokenizer, data):
    model.eval()
    output_data = []
    for entry in data:
        inputs = tokenizer(entry["maintext"], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # This should be improved depending on the type of answer extraction
        # Implementing only as a starting point for the task
        predicted_answer = tokenizer.decode(outputs[0][0].argmax(-1))
        entry["predicted_answer"] = predicted_answer
        output_data.append(entry)
    return output_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to the input file")
    parser.add_argument('--output', required=True, help="Path to the output file")
    args = parser.parse_args()

    data = read_input_file(args.input)
    tokenizer, model = load_model()
    output_data = inference(model, tokenizer, data)
    write_output_file(output_data, args.output)

if __name__ == "__main__":
    main()