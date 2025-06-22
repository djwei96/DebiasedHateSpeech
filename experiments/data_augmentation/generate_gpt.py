import os
import argparse
import pandas as pd

from toxigen.pretrained_classifiers import HateBERT, ToxDectRoBERTa
from toxigen.language_models import GPT3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_path", 
                        default='GroNLP/hateBERT', 
                        type=str, 
                        help="Path to the pretrained classifier model"
                        )
    parser.add_argument("--openai_engine", 
                        default='gpt-3.5-turbo', 
                        type=str, 
                        help="Name of the OpenAI engine to use for text generation"
                        )
    parser.add_argument("--openai_key", 
                        default='sk-YOUR-SECRET-KEY-HERE', 
                        type=str, 
                        help="OpenAI API key"
                        )
    parser.add_argument("--max_tokens", 
                        default=256, 
                        type=int,
                        help="Maximum number of tokens for the generated response"
                        )
    args = parser.parse_args()

    openai_endpoint_url = f"https://api.openai.com/v1/engines/{args.openai_engine}/completions"
    generator = GPT3(endpoint_url=args.openai_endpoint_url, apikey=args.openai_key)
    classifier = HateBERT(args.classifier_path)

    groups = ["asian", "chinese", "black", "latino", "mexican", "native_american", "middle_east", "immigrant", "jewish", "muslim", "women", "bisexual", "lgbtq", "mental_disability", "physical_disability"]

    results = []
    for hate_or_neutral in ["hate", "neutral"]:
        for group in groups:
            with open(f"TOXIGEN/prompts/{hate_or_neutral}_{group}_1k.txt", "r") as f:
                prompts = f.read().splitlines()
                for i in range(len(prompts)):
                    prompt = prompts[i]
                    response = generator(prompt, max_tokens=args.max_tokens)
                    prob = classifier.from_text(response)
                    results.append([response, prob])
    
    df = pd.DataFrame(results, columns=["text", "prob"])
    df['amateur_label'] = df['prob'].apply(lambda x: 1 if x > 50 else 0)
    df = df[['text', 'amateur_label']]
    df['id'] = range(len(df))
    df = df[['id', 'text', 'amateur_label']]
    df.to_json('gpt_data.jsonl', orient='records', lines=True)
