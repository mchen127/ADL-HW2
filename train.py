import os
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Adafactor
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from accelerate import Accelerator
from argparse import Namespace
# import sys
# sys.path.append(file_path_prefix)
from rouge_evaluation.tw_rouge.tw_rouge.twrouge import get_rouge


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
        title = item["title"]
        inputs = self.tokenizer(
            maintext,
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        labels = self.tokenizer(
            title,
            max_length=self.max_output_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
            "id": id,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for training MT5 model")

    # Add arguments
    parser.add_argument(
        "--model_name", type=str, default="google/mt5-small", help="Model name or path"
    )
    parser.add_argument(
        "--train_data_path", type=str, required=True, help="Path to the training data"
    )
    parser.add_argument(
        "--batch_size", type=int, default=24, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=8, help="Number of training epochs"
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
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=5,
        help="Number of steps for gradient accumulation",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation",
    )
    parser.add_argument(
        "--use_adafactor", action="store_true", help="Use Adafactor optimizer"
    )

    args = parser.parse_args()
    return args


def main(args):
    # Initialize Accelerator
    accelerator = Accelerator()

    # Load pre-trained multilingual T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)

    # Prepare dataset
    dataset = NewsSummaryDataset(
        filepath=args.train_data_path,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        # max_train=1000
    )

    # Split dataset into training and validation if required
    if args.validation_split:
        train_size = int((1 - args.validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )
    else:
        train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = None

    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Prepare model and optimizer
    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        (
            Adafactor(
                model.parameters(),
                # lr=args.learning_rate,
                scale_parameter=True,
                relative_step=True,
            )
            if args.use_adafactor
            else torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        ),
        train_dataloader,
    )

    # Set up training parameters
    device = accelerator.device

    # Tracking losses
    training_losses_per_batch = []
    validation_losses_per_epoch = []
    rouge_scores_per_epoch = []

    # Training loop with progress bar
    model.train()
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        epoch_loss = 0

        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            labels[labels == tokenizer.pad_token_id] = (
                -100
            )  # Ignore padding tokens in loss calculation

            assert torch.all(torch.isfinite(input_ids)), "input_ids contains NaN or Inf"
            assert torch.all(
                torch.isfinite(attention_mask)
            ), "attention_mask contains NaN or Inf"
            assert torch.all(torch.isfinite(labels)), "labels contain NaN or Inf"

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
            loss = (
                loss / args.gradient_accumulation_steps
            )  # Scale loss for gradient accumulation

            # Backward pass
            loss.backward()

            # Update weights and reset gradients if gradient accumulation step is complete
            if (i + 1) % args.gradient_accumulation_steps == 0:
                # Optional: Enable gradient clipping to stabilize training
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()  # Perform optimization step
                optimizer.zero_grad()  # Reset gradients after update

            # Track loss
            epoch_loss += loss.item()
            training_losses_per_batch.append(loss.item())
            progress_bar.set_postfix({"loss": loss.item()})
            if i < 5:
                print(
                    f"Epoch {epoch+1}, Batch {i+1}/{len(train_dataloader)}, Loss: {loss.item()}"
                )

        average_epoch_loss = epoch_loss / len(train_dataloader)

        # Validation loop
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            references = []
            hypotheses = []
            with torch.no_grad():
                for idx, batch in enumerate(val_dataloader):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    val_loss += outputs.loss.item()

                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=args.max_output_length,
                        num_beams=20,  # Use beam search with more beams for better generation
                        early_stopping=True,  # Stop when all beams have completed
                        temperature=1.0,  # Control randomness in generation
                        top_p=0.95,  # Use nucleus sampling with top-p
                        top_k=50,  # Or top-k sampling to limit the vocabulary
                    )

                    decoded_labels = tokenizer.batch_decode(
                        labels, skip_special_tokens=True
                    )
                    decoded_preds = tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )

                    references.extend(decoded_labels)
                    hypotheses.extend(decoded_preds)

            val_loss /= len(val_dataloader)
            validation_losses_per_epoch.append(val_loss)
            print(f"Validation Loss after epoch {epoch + 1}: {val_loss}")
            rouge_scores = get_rouge(hypotheses, references)
            rouge_scores_per_epoch.append(rouge_scores)
            print(f"ROUGE Scores after epoch {epoch + 1}: {rouge_scores}")
            model.train()

        # Save model and tokenizer
        state_dict = model.state_dict()
        for name, tensor in state_dict.items():
            if not tensor.is_contiguous():
                state_dict[name] = tensor.contiguous()
        torch.save(
            state_dict,
            f"./model/fine-tuned-mt5-small-epoch-{epoch+1}.pth",
        )
        tokenizer.save_pretrained(
            f"./tokenizer/fine-tuned-mt5-small-epoch-{epoch+1}",
        )

    # Training Loss per Batch
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(training_losses_per_batch) + 1),
        training_losses_per_batch,
        label="Training Loss",
        color="b",
    )
    plt.xlabel("Batches")
    plt.ylabel("Training Loss")
    plt.title("Training Loss per Batch")
    plt.legend()
    plt.savefig(f"./results/training_loss_per_batch.png")
    plt.show()

    # Validation Loss per Epoch
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(validation_losses_per_epoch) + 1)
    plt.plot(epochs, validation_losses_per_epoch, label="Validation Loss", color="r")
    plt.scatter(epochs, validation_losses_per_epoch, color="r", s=15)  # Show data points

    # Annotate each point with its actual value
    for x, y in zip(epochs, validation_losses_per_epoch):
        plt.text(x, y + 0.02, f"{y:.3f}", fontsize=9, ha="center", va="bottom")

    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss per Epoch")
    plt.legend()
    plt.savefig(f"./results/validation_loss_per_epoch.png")
    plt.show()

    # ROUGE Scores per Epoch
    rouge_1_f1 = [score["rouge-1"]["f"] for score in rouge_scores_per_epoch]
    rouge_2_f1 = [score["rouge-2"]["f"] for score in rouge_scores_per_epoch]
    rouge_l_f1 = [score["rouge-l"]["f"] for score in rouge_scores_per_epoch]

    epochs = range(1, len(rouge_1_f1) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rouge_1_f1, label="ROUGE-1 F1", color="g")
    plt.scatter(epochs, rouge_1_f1, color="g", s=15)  # Show data points for ROUGE-1
    for x, y in zip(epochs, rouge_1_f1):
        plt.text(x, y + 0.003, f"{y:.3f}", fontsize=9, ha="center", va="bottom")

    plt.plot(epochs, rouge_2_f1, label="ROUGE-2 F1", color="c")
    plt.scatter(epochs, rouge_2_f1, color="c", s=15)  # Show data points for ROUGE-2
    for x, y in zip(epochs, rouge_2_f1):
        plt.text(x, y + 0.003, f"{y:.3f}", fontsize=9, ha="center", va="bottom")

    plt.plot(epochs, rouge_l_f1, label="ROUGE-L F1", color="m")
    plt.scatter(epochs, rouge_l_f1, color="m", s=15)  # Show data points for ROUGE-L
    for x, y in zip(epochs, rouge_l_f1):
        plt.text(x, y + 0.003, f"{y:.3f}", fontsize=9, ha="center", va="bottom")

    plt.xlabel("Epochs")
    plt.ylabel("ROUGE Score (F1)")
    plt.title("ROUGE Scores per Epoch")
    plt.legend()
    plt.savefig(f"./results/rouge_scores_per_epoch.png")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)
