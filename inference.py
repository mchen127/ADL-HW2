# Evaluation loop on the evaluation dataset
import json
from tqdm import tqdm
import os
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    MT5ForConditionalGeneration,
    MT5Tokenizer,
)
from torch.utils.data import Dataset, DataLoader
import argparse
from accelerate import Accelerator


# Define dataset class
class NewsSummaryDataset(Dataset):
    def __init__(
        self,
        filepath,
        tokenizer,
        max_input_length=256,
        max_output_length=64,
        max_train=-1,
    ):
        self.data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_train != -1 and idx >= max_train:
                    break
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        id = item["id"]
        maintext = item["maintext"]
        # title = item["title"]
        inputs = self.tokenizer(
            maintext,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # labels = self.tokenizer(
        #     title,
        #     max_length=self.max_output_length,
        #     truncation=True,
        #     padding="max_length",
        #     return_tensors="pt",
        # )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            # "labels": labels["input_ids"].squeeze(),
            "id": id,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments for MT5 text summarization model"
    )

    # Add arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="./google_mt5_small",
        help="Model name or path",
    )
    parser.add_argument(
        "--eval_dataset_path",
        type=str,
        required=True,
        help="Path to the evaluation dataset",
    )
    parser.add_argument(
        "--submission_path",
        type=str,
        required=True,
        help="Path to save the submission file",
    )
    parser.add_argument(
        "--fine_tuned_checkpoint_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=512,
        help="Maximum input sequence length",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=100,
        help="Maximum output sequence length",
    )

    # New arguments for alternate beam search and generation settings
    parser.add_argument(
        "--num_beams", type=int, default=20, help="Number of beams for beam search"
    )
    parser.add_argument(
        "--early_stopping", action="store_true", help="Whether to stop decoding early"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.95, help="Nucleus sampling probability"
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    args = parser.parse_args()
    return args


def main(args):
    # Initialize Accelerator
    accelerator = Accelerator()

    # Prepare model and optimizer
    # Load pre-trained multilingual T5 model and tokenizer
    tokenizer = MT5Tokenizer.from_pretrained(args.tokenizer_path)
    # Initialize the model with the base architecture
    model = MT5ForConditionalGeneration.from_pretrained(args.model_path)

    # Load the fine-tuned weights
    model.load_state_dict(torch.load(args.fine_tuned_checkpoint_path))

    eval_dataset = NewsSummaryDataset(
        filepath=args.eval_dataset_path,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
    )

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False
    )

    model, eval_dataloader = accelerator.prepare(
        model,
        eval_dataloader,
    )

    device = accelerator.device
    # state_dict = torch.load(args.model_path, map_location=device)
    # model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    results = []
    # Add progress bar for evaluation
    with torch.no_grad():
        progress_bar = tqdm(eval_dataloader, desc="Evaluating")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = batch["id"]

            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=args.max_output_length,
                num_beams=args.num_beams,
                early_stopping=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )

            # Decode predictions
            decoded_preds = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            # Format each prediction and add it to results
            for i, title in enumerate(decoded_preds):
                result = {"title": title, "id": ids[i]}
                results.append(result)

    # Write results to the submission file in the required format
    with open(args.submission_path, "w", encoding="utf-8") as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")

    print(f"Submission file saved to: {args.submission_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # print(args)  # You can remove this line; it's for debugging purposes
