# ADL-HW2 MT5 Text Summarization Model
This repository contains a training script for fine-tuning the MT5 model for text summarization using a custom dataset. The script supports multilingual T5 models and includes features such as gradient accumulation, Adafactor optimization, and ROUGE evaluation.


## Environment
`pip install -r requirements.txt`

## Training
I provided two ways to train, Google colab or if you have GPU.

### Dataset
The training script expects a dataset in .jsonl format, where each line contains a JSON object with the following structure:

```json
{
    "id": "unique_id",
    "maintext": "The main text of the article or input sequence",
    "title": "The summary or target sequence"
}
```
### Google Colab Training
* Run `train.ipynb`
* Make sure the files you need are in your drive, and mount to the notebook.
Execute every code block to train.
* You can modify the hyperparameters in the "args" code block.
* The code will save models from every epoch and plot 3 graphs training loss per batch, validation loss per batch, and ROUGE score per epoch for you.
### Normal Training
* Run `train.py`
* In normal training, You can modify the hyperparameters by passing args when executing the file.
* The code will also save models from every epoch and plot 3 graphs training loss per batch, validation loss per batch, and ROUGE score per epoch for you.
```bash
python train.py \
    --train_data_path /path/to/train.jsonl \ 
    --batch_size 16 \
    --learning_rate 5e-5 \ 
    --epochs 8 \
    --max_input_length 512 \
    --max_output_length 100 \
    --gradient_accumulation_steps 5 \
    --validation_split 0.1 \
    --use_adafactor
```
### Command-line Arguments

| Argument                     | Description                                                         | Default Value       |
|-------------------------------|---------------------------------------------------------------------|---------------------|
| `--model_name`                | Name or path of the pretrained model (e.g., `google/mt5-small`)      | `google/mt5-small`  |
| `--train_data_path`           | Path to the training data in `.jsonl` format                        | **Required**        |
| `--batch_size`                | Batch size for training                                             | 24                  |
| `--learning_rate`             | Learning rate for the optimizer                                     | 5e-5                |
| `--epochs`                    | Number of training epochs                                           | 8                   |
| `--max_input_length`          | Maximum input sequence length                                       | 512                 |
| `--max_output_length`         | Maximum output sequence length                                      | 100                 |
| `--gradient_accumulation_steps`| Number of steps for gradient accumulation                           | 5                   |
| `--validation_split`          | Fraction of the dataset to use for validation                       | 0.1                 |
| `--use_adafactor`             | Use Adafactor optimizer instead of AdamW                            | Disabled by default |

### Output
* Models:\
  The models will be saved in `./model`
* Tokenizers\
  The tokenizers will be saved in `./tokenizer`
* Results\
  The training result images will be saved in `./results`
