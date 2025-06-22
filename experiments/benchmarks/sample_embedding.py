import os
import torch
import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans


def get_embeddings(texts, tokenizer, model):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return np.array(embeddings)


def sample_from_embeddings(df, model_name, sample_num):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = get_embeddings(df['processed_text'].tolist(), tokenizer, model)
    kmeans = KMeans(n_clusters=sample_num)
    df['cluster'] = kmeans.fit_predict(embeddings)
    sample_df = df.groupby('cluster').apply(lambda x: x.sample(1)).reset_index(drop=True)
    sample_df.drop('cluster', axis=1, inplace=True)
    return sample_df


def main(args):
    df = pd.read_json(os.path.join(args.data_dir, 'processed_data.jsonl'), lines=True)
    sample_df = sample_from_embeddings(df, args.model_name, args.sample_num)
    return sample_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='', 
                        help='Directory to load the data'
                        )
    parser.add_argument('--sample_num', 
                        type=int, 
                        default=256, 
                        help='Number of samples'
                        )
    parser.add_argument('--model_name', 
                        type=str, 
                        default='bert-base-uncased', 
                        help='Name of the model'
                        )

    args = parser.parse_args()
    sample_df = main(args)
