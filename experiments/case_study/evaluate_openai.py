import os
import json
import time
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from openai import OpenAI


def main(args):
    client = OpenAI(api_key=args.api_key)
    df_test = pd.read_json(os.path.join(args.data_dir, f'test.jsonl'), lines=True)

    all_labels = []
    for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
        question = args.prompt.format(row['processed_text'])
        messages = [{"role": "system", "content": "You are a helpful hate speech detector."},
                   {"role": "user", "content": question}]
        
        response = client.chat.completions.create(model=args.model_name, messages=messages)
        response = response.choices[0].message.content
        if 'yes' in response.lower():
            label = 1
        else:
            label = 0
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
                        default='gpt-4o',
                        help='Name of the fine-tuned model'
                        )
    parser.add_argument('--api_key',
                        type=str,
                        default='YOUR-SECRET-KEY-HERE',
                        help='OpenAI API key'
                        )
    parser.add_argument('--prompt',
                        type=str,
                        default='Is the following speech hateful? Answer Yes or No. \"{}\"',
                        help='Prompt for the model'
                        )

    args = parser.parse_args()
    main(args)