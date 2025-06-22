import os
import argparse
import pandas as pd
from detector.utils import TextPreprocessor


def main(args):
    file_path = os.path.join(args.data_dir, args.file_name)
    df = pd.read_json(file_path, lines=True)

    processor = TextPreprocessor()
    df['processed_text'] = df['text'].apply(processor.preprocess_text)
    file_name = args.file_name.replace('raw', 'processed')
    df.to_json(os.path.join(args.data_dir, file_name), lines=True, orient='records')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True, 
        help='Directory containing the data file'
        )
    parser.add_argument(
        '--file_name', 
        type=str, 
        required=True, 
        help='Name of the data file'
        )

    args = parser.parse_args()
    main(args)
 