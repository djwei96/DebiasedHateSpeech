import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer


class ContrastiveDebiasingDataset(Dataset):
    def __init__(self, texts, labels, protected_attributes, tokenizer, max_seq_length):
        self.texts = texts
        self.labels = labels
        self.protected_attributes = protected_attributes
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]
        protected_attribute = self.protected_attributes[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'protected_attribute': torch.tensor(protected_attribute, dtype=torch.long)
        }


class ContrastiveDebiasingModel(nn.Module):
    def __init__(self, model_name, hidden_size=768):
        super(ContrastiveDebiasingModel, self).__init__()
        self.plm = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(hidden_size, 2)
        self.temperature = 0.07

    def forward(self, input_ids, attention_mask):
        outputs = self.plm(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state[:, 0, :]
        logits = self.fc(sequence_output)
        return logits, sequence_output

    def contrastive_loss(self, embeddings, labels, protected_attribute):
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        sim_matrix = F.softmax(sim_matrix, dim=1)
        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        negative_mask = 1 - positive_mask
        
        positive_sim = sim_matrix * positive_mask
        negative_sim = sim_matrix * negative_mask
        contrastive_loss = -torch.log(positive_sim.sum(dim=1) / (negative_sim.sum(dim=1) + 1e-10))
        
        protected_mask = (protected_attribute.unsqueeze(1) == protected_attribute.unsqueeze(0)).float()
        fair_contrastive_loss = -torch.log(protected_mask.sum(dim=1) / (negative_sim.sum(dim=1) + 1e-10))
        
        total_loss = contrastive_loss.mean() + fair_contrastive_loss.mean()
        return total_loss


def evaluate(model, text, tokenizer, max_seq_length, device):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        logits, _ = model(input_ids, attention_mask)
    
    label = torch.argmax(logits, dim=1)
    return label.item()


def train(model, dataloader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            protected_attribute = batch['protected_attribute'].to(device)
            
            optimizer.zero_grad()
            logits, embeddings = model(input_ids, attention_mask)
            loss_ce = criterion(logits, labels)
            contrastive_loss = model.contrastive_loss(embeddings, labels, protected_attribute)
            total_loss = loss_ce + contrastive_loss
            total_loss.backward()
            optimizer.step()


def create_dataloader(df_train, tokenizer, args):
    train_dataset = ContrastiveDebiasingDataset(
        texts=df_train['processed_text'].values,
        labels=df_train['expert_label'].values,
        protected_attributes=df_train['group'].map({'aae': 0, 'white': 1, 'others': 2}).values,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length
    )
    return DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)


def main(args):
    if args.use_cuda:
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    else:
        device = 'cpu'

    df_train = pd.read_json(os.path.join(args.data_dir, 'train.jsonl'), lines=True)
    df_test = pd.read_json(os.path.join(args.data_dir, 'test.jsonl'), lines=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataloader = create_dataloader(df_train, tokenizer, args)

    model = ContrastiveDebiasingModel(model_name=args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train(model, train_dataloader, optimizer, criterion, epochs=args.num_epochs, device=device)

    df_test = df_test.head(10)
    all_results = []
    for text in df_test['processed_text'].to_list():
        label = evaluate(model, text, tokenizer, args.max_seq_length, device)
        all_results.append(label)
    print(all_results)
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='', 
                        help='Directory to load the data'
                        )
    parser.add_argument('--use_cuda', 
                        type=bool, 
                        default=True, 
                        help='Whether to use cuda'
                        )
    parser.add_argument('--cuda_visible_devices', 
                        type=str, 
                        default='5',
                        help='CUDA visible devices'
                        )
    parser.add_argument('--train_batch_size', 
                        type=int, 
                        default=32,
                        help='Batch size for training'
                        )
    parser.add_argument('--max_seq_length', 
                        type=int, 
                        default=128,
                        help='Max sequence length'
                        )
    parser.add_argument('--num_epochs', 
                        type=int, 
                        default=15,
                        help='Number of epochs'
                        )
    parser.add_argument('--model_name', 
                        type=str, 
                        default='bert-base-uncased',
                        help='Name of the model'
                        )
    parser.add_argument('--lr', 
                        type=float, 
                        default=5e-5,
                        help='Learning rate'
                        )

    args = parser.parse_args()
    all_results = main(args)
