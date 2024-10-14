import json
import argparse
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from rouge_evaluation.tw_rouge.tw_rouge.twrouge import get_rouge


def generate_summary(model, tokenizer, input_text, generation_config):
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=generation_config["max_input_length"],
    ).input_ids.to(model.device)

    # Generate summary
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_length=generation_config["max_output_length"],
            num_beams=generation_config["num_beams"],
            do_sample=generation_config["do_sample"],
            top_k=generation_config["top_k"],
            top_p=generation_config["top_p"],
            temperature=generation_config["temperature"],
            repetition_penalty=generation_config["repetition_penalty"],
            length_penalty=generation_config["length_penalty"],
            early_stopping=generation_config["early_stopping"],
        )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def evaluate_summary(hypotheses, references):
    rouge_scores = get_rouge(hypotheses, references)
    return rouge_scores


def main(args):
    # Load the model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load input data
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    # Generation configuration
    generation_config = {
        "max_input_length": args.max_input_length,
        "max_output_length": args.max_output_length,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "early_stopping": args.early_stopping,
    }

    # Generate summaries and evaluate
    hypotheses = []
    references = []
    for item in data:
        maintext = item["maintext"]
        reference_title = item["title"]

        generated_summary = generate_summary(
            model, tokenizer, maintext, generation_config
        )
        hypotheses.append(generated_summary)
        references.append(reference_title)

        print(f"Input: {maintext}")
        print(f"Generated Summary: {generated_summary}")
        print(f"Reference Title: {reference_title}")
        print("\n")

    # Evaluate using ROUGE
    rouge_scores = evaluate_summary(hypotheses, references)
    print("ROUGE Scores:", rouge_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/mt5-small",
        help="Pre-trained model name or path",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input data in JSONL format",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=256,
        help="Maximum input length for tokenization",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=64,
        help="Maximum output length for generation",
    )
    parser.add_argument(
        "--num_beams", type=int, default=4, help="Number of beams for beam search"
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling; use in conjunction with top_k or top_p",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="The number of highest probability vocabulary tokens to keep for top-k sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The value used to module the next token probabilities",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="The parameter for repetition penalty",
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=1.0,
        help="Exponential penalty to the length",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Whether to stop the beam search when at least num_beams sentences are finished",
    )

    args = parser.parse_args()
    main(args)
