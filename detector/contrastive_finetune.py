import os
import json
import argparse
import subprocess
import pandas as pd
from datetime import datetime


def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join(args.save_dir, f"{timestamp}")
    os.makedirs(result_dir)

    df = pd.read_json(os.path.join(args.data_dir, 'train.jsonl'), lines=True)
    df = df[['processed_text', 'generated_pair']]

    data_path = os.path.join(result_dir, 'pairs.csv')
    df.to_csv(os.path.join(data_path), index=False)

    with open(args.config_path, 'r') as f:
        config = json.load(f)
    config['train_file'] = os.path.abspath(data_path)

    model_dir = os.path.join(result_dir, 'model')
    model_dir = os.path.abspath(model_dir)
    config['output_dir'] = model_dir 

    config_path = os.path.join(result_dir, 'config.json')
    config_path = os.path.abspath(config_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    os.chdir(args.simcse_dir)
    subprocess.run(['python', 'train.py', config_path])
    subprocess.run(['python', 'simcse_to_huggingface.py', '--path', model_dir])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        type=str, 
                        required=True, 
                        help='Directory to load the data'
                        )
    parser.add_argument('--simcse_dir', 
                        type=str, 
                        required=True, 
                        help='Directory to the simcse repo'
                        )
    parser.add_argument('--config_path', 
                        type=str, 
                        required=True, 
                        help='Path to the config file of contrastive finetuning'
                        )
    parser.add_argument('--save_dir', 
                        type=str, 
                        required=True, 
                        help='Directory to save the model'
                        )

    args = parser.parse_args()
    main(args)
