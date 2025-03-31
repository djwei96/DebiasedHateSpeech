import os
import pandas as pd
from datasets import load_dataset


if __name__ == '__main__':
    '''
    1. Fill out the access form at https://forms.office.com/r/r6VXX8f8vh to get authorization token
    2. Download the data from this https://huggingface.co/datasets/toxigen/toxigen-data with authorization token
    3. Select the subset named "annotated" and the "train" split.
    4. Save the data as a CSV file.
    5. Run the following code to transform the data into the required format.
    '''
    df = pd.read_csv('hf_cache/data.csv')
    df = df[['text', 'toxicity_ai']].dropna()
    df['amateur_label'] = df['toxicity_ai'].apply(lambda x: 1 if x > 2.5 else 0)
    df['id'] = range(len(df))
    df = df[['id', 'text', 'amateur_label']]
    df.to_json('toxigen_data.jsonl', orient='records', lines=True)
