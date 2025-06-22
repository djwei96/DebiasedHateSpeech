import os
import argparse
import pandas as pd
from googleapiclient import discovery


def main(args):
    df = pd.read_json(os.path.join(args.data_dir, 'test.jsonl'), lines=True)
    client = discovery.build("commentanalyzer",
                             "v1alpha1",
                             developerKey=args.api_key,
                             discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                             static_discovery=False,
                             )
    all_labels = []
    for _, row in df.iterrows():
        analyze_request = {
            'comment': {'text': row['processed_text']},
            'requestedAttributes': {'TOXICITY': {}}
        }
        response = client.comments().analyze(body=analyze_request).execute()
        score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        label = 1 if score > args.threshold else 0
        all_labels.append(label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze text toxicity using Perspective API.")
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='', 
                        help='Directory to load the data'
                        )
    parser.add_argument('--api_key',
                        type=str,
                        default='YOUR-SECRET-KEY-HERE',
                        help='Perspective API key'
                        )
    parser.add_argument('--threshold', 
                        type=float, 
                        default=0.5, 
                        help='Threshold for Perspective API.')
    args = parser.parse_args()


    if __name__ == '__main__':
        main(args)
