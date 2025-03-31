import os
import json
import torch
import stanza
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

from difflib import SequenceMatcher
from simcse import SimCSE


class DataSampler:
    def __init__(self, df, group_col, score_col):
        self.df = df
        self.group_col = group_col
        self.score_col = score_col

    def _calculate_sample_counts(self, n):
        group_counts = self.df[self.group_col].value_counts()
        group_ratios = group_counts / group_counts.sum()

        exact_counts = group_ratios * n
        int_counts = exact_counts.astype(int)
        decimal_part = exact_counts - int_counts

        remaining_samples = n - int_counts.sum()
        adjustment_indices = np.argsort(-decimal_part.values)[:remaining_samples]
        adjusted_counts = int_counts.copy()
        adjusted_counts.iloc[adjustment_indices] += 1

        return adjusted_counts

    def sample(self, n):
        sample_counts = self._calculate_sample_counts(n)

        train_dfs = [
            self.df[self.df[self.group_col] == group]
            .nlargest(sample_counts[group], self.score_col)
            for group in sample_counts.index
        ]

        train_df = pd.concat(train_dfs)
        test_df = self.df[~self.df.index.isin(train_df.index)]

        return train_df, test_df


class HateTargetDetector:
    def __init__(self, hate_target_vocab_path, threshold=0.8):
        self.hate_target_vocab = self._load_hate_target_vocab(hate_target_vocab_path)
        self.threshold = threshold

    @staticmethod
    def _load_hate_target_vocab(hate_target_vocab_path):
        with open(hate_target_vocab_path, "r") as f:
            hate_target_vocab = json.load(f)
            return hate_target_vocab

    @staticmethod
    def _calculate_similarity(text1, text2):
        return SequenceMatcher(None, text1, text2).ratio()

    def _edit_distance_match(self, text):
        words = text.split()

        for target, target_words in self.hate_target_vocab.items():
            for target_word in target_words:
                for word in words:
                    if self._calculate_similarity(target_word, word) >= self.threshold:
                        return target, word
        return None, None

    def _direct_match(self, text):
        words = text.split()

        for target, target_words in self.hate_target_vocab.items():
            for target_word in target_words:
                for word in words:
                    if target_word in word:
                        return target, word
        
        return None, None

    def detect_hate_target(self, text):
        hate_target, hate_word = self._direct_match(text)
        if not hate_target:
            hate_target, hate_word = self._edit_distance_match(text)

        return {'hate target': hate_target, 'hate word': hate_word}


class PairGenerator:
    def __init__(self, template_path, stanza_dir='stanza_resources', model_name='princeton-nlp/sup-simcse-roberta-base', device='cuda'):
        self.templates = self._load_templates(template_path)
        self.nlp = self._initialize_dependency_parser(dir=stanza_dir)
        self.candidate_ranker = SimCSE(model_name, device=device)

    @staticmethod
    def _load_templates(template_path):
        with open(template_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def _initialize_dependency_parser(dir):
        return stanza.Pipeline(dir=dir, lang="en", processors="tokenize,mwt,pos,lemma,depparse,ner")

    def extract_subject(self, text):
        doc = self.nlp(text)

        for sentence in doc.sentences:
            for word in sentence.words:
                if word.deprel == "nsubj":
                    subject = word.text.lower()

                    for ent in sentence.entities:
                        if ent.text.lower() == subject and ent.type == "PERSON":
                            return subject
        return None

    def generate_candidates(self, text, label, detect_results):
        hate_target = detect_results.get('hate target') or "human"
        hate_word = detect_results.get('hate word')

        subject = self.extract_subject(text) or "i"
        if subject == hate_word or (len(subject) == 1 and subject != "i"):
            subject = "i"

        candidates = [
            t.replace("**subject**", subject).replace("**object**", hate_target)
            for t in (self.templates['hateful'] if label == 1 else self.templates['unhateful'])
            ]

        return self.rank_candidates(text, candidates)

    def rank_candidates(self, text, candidates, topk=1):
        similarities = self.candidate_ranker.similarity(text, candidates)
        similarities = torch.tensor(similarities.flatten())
        top_indices = torch.topk(similarities, topk).indices.cpu().tolist()
        return candidates[top_indices[0]]


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_dir = os.path.join(args.save_dir, f"{timestamp}")
    os.makedirs(result_dir)

    #stanza.download('en', model_dir=args.stanza_dir, processors='tokenize,mwt,pos,lemma,depparse,ner')

    data_path = os.path.join(args.data_dir, 'processed_data.jsonl')
    df = pd.read_json(data_path, lines=True)

    data_sampler = DataSampler(df, group_col='group', score_col='dialect_score')
    train_df, test_df = data_sampler.sample(args.sample_num)

    detector = HateTargetDetector(args.hate_target_vocab_path)
    generator = PairGenerator(args.template_path, stanza_dir=args.stanza_dir, model_name=args.model_name, device=args.device)

    for index, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        text = row['processed_text']
        label = row['expert_label']

        detect_results = detector.detect_hate_target(text)
        train_df.loc[index, 'hate_target'] = detect_results['hate target']

        best_candidate = generator.generate_candidates(text, label, detect_results)
        train_df.loc[index, 'generated_pair'] = best_candidate

    for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        text = row['processed_text']
        detect_results = detector.detect_hate_target(text)
        test_df.loc[index, 'hate_target'] = detect_results['hate target']

    train_df.to_json(os.path.join(result_dir, 'train.jsonl'), lines=True, orient='records')
    test_df.to_json(os.path.join(result_dir, 'test.jsonl'), lines=True, orient='records')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_path', 
                        type=str, 
                        required=True, 
                        help='Path to the pair template file'
                        )
    parser.add_argument('--hate_target_vocab_path', 
                        type=str, 
                        required=True, 
                        help='Path to the hate target vocabulary file'
                        )
    parser.add_argument('--data_dir', 
                        type=str, 
                        required=True, 
                        help='Directory containing the data files'
                        )
    parser.add_argument('--stanza_dir', 
                        type=str, 
                        required=True, 
                        help='Directory for the stanza package'
                        )
    parser.add_argument('--save_dir', 
                        type=str, 
                        required=True, 
                        help='Directory for saving the train and test data'
                        )
    parser.add_argument('--model_name', 
                        type=str, 
                        required=True, 
                        help='Name of the model to use for candidate ranking'
                        )
    parser.add_argument('--device', 
                        type=str, 
                        required=True, 
                        help='Device to run the model on'
                        )
    parser.add_argument('--cuda_visible_devices', 
                        type=str, 
                        required=True, 
                        help='CUDA visible devices'
                        )
    parser.add_argument('--seed', 
                        type=int, 
                        required=True, 
                        help='Random seed'
                        )
    parser.add_argument('--sample_num', 
                        type=int, 
                        required=True, 
                        help='Number of samples to generate'
                        )

    args = parser.parse_args()
    main(args)
