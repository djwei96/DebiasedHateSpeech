import os 
import re
import json
import string
import numpy as np
import pandas as pd

from openprompt.data_utils.data_processor import DataProcessor
from openprompt.data_utils.utils import InputExample


class TextPreprocessor:
    def __init__(self):
        self.contractions = {
            " cant ": " can't ",
            " dont ": " don't ",
            " im ": " i'm ",
            " wud ": " would ",
            " cud ": " could ",
            " shud ": " should ",
            " u ": " you ",
            " ur ": " your ",
            " r ": " are ",
            " y ": " why ",
            " b4 ": " before ",
            " thx ": " thanks ",
            " pls ": " please ",
        }
        self.default_replacements = {
            "$": "s",
            "@": "a",
            "33": "ee",
            "m3n": "men",
            "lik3": "like",
            "@flotu": " ",
            "@realdonaldtrump": " ",
            "@rishisunak": " ",
        }

    def preprocess_text(self, text):
        text = text.encode('utf-8').decode('utf-8')
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'rt\s+@[\w]+:', ' ', text)
        text = ' '.join(text.strip().split())
        for contraction, replacement in self.contractions.items():
            text = text.replace(contraction, replacement)

        for key, value in self.default_replacements.items():
            text = text.replace(key, value)

        text = re.sub(f'[^a-zA-Z0-9{re.escape(string.punctuation)}]', ' ', text)
        text = ' '.join(text.strip().split())
        text = re.sub(r'(\W)\1+', r'\1', text)

        for char in string.punctuation:
            text = text.replace(char, f' {char} ')

        text = ' '.join(text.strip().split())

        return text


class OpenPromptProcessor(DataProcessor):
    def __init__(self):
        super().__init__()

    def get_examples(self, data_dir, split):
        df = self._load_data(data_dir, split)
        return self._convert_to_examples(df, split)

    def _load_data(self, data_dir, split):
        file_path = os.path.join(data_dir, f"{split}.jsonl")
        try:
            return pd.read_json(file_path, lines=True)
        except Exception as e:
            raise ValueError(f"Error loading JSONL file: {file_path}") from e

    def _convert_to_examples(self, df, split):
        examples = []
        for _, row in df.iterrows():
            examples.append(self._create_example(row, split))
        return examples

    def _create_example(self, row, split):
        text_a = row["processed_text"]
        label = int(row["expert_label"])
        target = row["hate_target"] if pd.notna(row["hate_target"]) else "no one"

        guid = f"{split}-{row['id']}"
        meta = {"amateur_label": row["amateur_label"], "target": target}

        return InputExample(guid=guid, text_a=text_a, label=label, meta=meta)

