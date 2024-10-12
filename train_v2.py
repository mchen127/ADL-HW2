import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Adafactor
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt


# Define dataset class
class NewsSummaryDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_input_length=256, max_output_length=64):
        self.data = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
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
            "labels": labels["input_ids"].squeeze().contiguous(),
        }


# Main function
def main(args):
    # Load pre-trained multilingual T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Prepare dataset
    dataset = NewsSummaryDataset(
        filepath=args.data_path,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
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

    # Set up training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = (
        Adafactor(
            model.parameters(),
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
        )
        if args.use_adafactor
        else torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    )

    # Mixed precision training
    scaler = torch.amp.GradScaler("cuda") if args.use_fp16 else None

    # Tracking losses
    training_losses = []
    validation_losses = []

    # Training loop with progress bar
    model.train()
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        optimizer.zero_grad()
        epoch_loss = 0
        for i, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).contiguous()

            with torch.amp.autocast("cuda", enabled=args.use_fp16):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps

            if args.use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                if args.use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.gradient_accumulation_steps
            # print(loss.item())
            progress_bar.set_postfix(
                {"loss": loss.item() * args.gradient_accumulation_steps}
            )

        average_epoch_loss = epoch_loss / len(train_dataloader)
        training_losses.append(average_epoch_loss)

        # Validation loop
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device).contiguous()

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    val_loss += outputs.loss.item()
            val_loss /= len(val_dataloader)
            validation_losses.append(val_loss)
            print(f"Validation Loss after epoch {epoch + 1}: {val_loss}")
            model.train()

    # Save the fine-tuned model
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)

    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss")
    if validation_losses:
        plt.plot(
            range(1, len(validation_losses) + 1),
            validation_losses,
            label="Validation Loss",
        )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("training_validation_loss.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/mt5-small",
        help="Pre-trained model name or path",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the training data in JSONL format",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="fine_tuned_mt5",
        help="Path to save the fine-tuned model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=256,
        help="Maximum sequence length for input tokenization",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=64,
        help="Maximum sequence length for output tokenization",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating weights",
    )
    parser.add_argument(
        "--use_fp16",
        action="store_true",
        help="Use 16-bit floating point precision for training",
    )
    parser.add_argument(
        "--use_adafactor",
        action="store_true",
        help="Use Adafactor optimizer instead of AdamW",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Fraction of data to use for validation (0 means no validation set)",
    )

    args = parser.parse_args()
    main(args)
