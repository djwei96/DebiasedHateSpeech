import os
import json
import argparse
import pandas as pd


def text_matching(text, group, hate_target_vocab):
    label = 0
    for hate_target, hate_words in hate_target_vocab.items():
        for hate_word in hate_words:
            if hate_word in text:
                label = 1
                if group == 'aae' and hate_target == 'black people':
                    label = 0
                return label
    return label


def main(args):
    with open(args.hate_target_vocab_path, 'r') as f:
        hate_target_vocab = f.read()
        hate_target_vocab = json.loads(hate_target_vocab)
    
    df_test = pd.read_json(os.path.join(args.data_dir, 'test.jsonl'), lines=True)
    all_results = []
    for index, row in df_test.iterrows():
        text = row['processed_text']
        group = row['group']
        label = text_matching(text, group, hate_target_vocab)
        all_results.append(label)
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hate_target_vocab_path', 
                        type=str, 
                        default='datasets/hate_target_vocab.json', 
                        help='Path to the hate target vocabulary'
                        )
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='', 
                        help='Directory to load the data')

    args = parser.parse_args()
    main(args)
