import os
import torch
import argparse
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset as TorchDataset


class HuggingfaceDataset(TorchDataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocess_data(tokenizer, data, text_column, max_seq_length):
    return tokenizer(
        data[text_column].tolist(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length
    )


def evaluate(model, tokenizer, text, max_seq_length, device='cpu'):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_seq_length
    )
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    label = torch.argmax(logits, dim=-1).item()
    return label


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    df_train = pd.read_json(os.path.join(args.data_dir, 'train.jsonl'), lines=True)
    df_test = pd.read_json(os.path.join(args.data_dir, 'test.jsonl'), lines=True)
    df_train = df_train.sample(frac=0.1).reset_index(drop=True)
    df_test = df_test.head(5)
    cache_dir = '/mnt/Data/dongjun/huggingface_models'
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_labels = []
    for text in df_test['processed_text'].tolist():
        label = evaluate(model, tokenizer, text, args.max_seq_length)
    all_labels.append(label)
    return all_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='', 
                        help='Directory to load the data'
                        )
    parser.add_argument('--model_name', 
                        type=str, 
                        default='bert-base-uncased',
                        help='Name of the model'
                        )
    parser.add_argument('--max_seq_length', 
                        type=int, 
                        default=128,
                        help='Max sequence length'
                        )
    parser.add_argument('--num_epochs', 
                        type=int, 
                        default=2,
                        help='Number of epochs'
                        )
    parser.add_argument('--train_batch_size', 
                        type=int, 
                        default=4,
                        help='Batch size for training'
                        )
    parser.add_argument('--lr', 
                        type=float, 
                        default=5e-5, 
                        help='Learning rate'
                        )

    args = parser.parse_args()
    main(args)
