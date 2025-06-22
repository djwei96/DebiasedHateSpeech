import os
import argparse
import pandas as pd



def main(args):
    df = pd.read_json(os.path.join(args.data_dir, 'processed_data.jsonl'), lines=True)
    sample_df = df.sample(args.sample_num)
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

    args = parser.parse_args()
    train_df = main(args)
