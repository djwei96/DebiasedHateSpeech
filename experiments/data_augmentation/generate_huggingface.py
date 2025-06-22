import os
import argparse
import pandas as pd

from transformers import pipeline
from toxigen.pretrained_classifiers import HateBERT, ToxDectRoBERTa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier_path", 
                        default='GroNLP/hateBERT', 
                        type=str,
                        help="Path to the pretrained classifier model"
                        )
    parser.add_argument("--model_name", 
                        default='openai-community/gpt2', 
                        #default='meta-llama/Llama-3.3-70B-Instruct', # requires transformers>=4.46.0
                        type=str, 
                        help="Name of the model to use for text generation"
                        )
    parser.add_argument("--cache_dir", 
                        default='/mnt/Data/dongjun/DebiasHateSpeech/data_augmentation/hf_cache', 
                        type=str, 
                        help="Path to the cache directory for the model"
                        )
    parser.add_argument("--cuda_visible_devices", 
                        default='3', 
                        type=str, 
                        help="CUDA visible devices"
                        )
    parser.add_argument("--max_tokens", 
                        default=256, 
                        type=int, 
                        help="Maximum number of tokens for the generated response"
                        )
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    generator = pipeline("text-generation", model=args.model_name, cache_dir=args.cache_dir)
    classifier = HateBERT(args.classifier_path)

    groups = ["asian", "chinese", "black", "latino", "mexican", "native_american", "middle_east", "immigrant", "jewish", "muslim", "women", "bisexual", "lgbtq", "mental_disability", "physical_disability"]

    results = []
    for hate_or_neutral in ["hate", "neutral"]:
        for group in groups:
            with open(f"TOXIGEN/prompts/{hate_or_neutral}_{group}_1k.txt", "r") as f:
                prompts = f.read().splitlines()
                for i in range(len(prompts)):
                    prompt = prompts[i]
                    response = generator(prompt, max_length=args.max_tokens)
                    response = response[0]['generated_text']
                    prob = classifier.from_text(response)
                    results.append([response, prob])
    
    df = pd.DataFrame(results, columns=["text", "prob"])
    df['amateur_label'] = df['prob'].apply(lambda x: 1 if x > 50 else 0)
    df = df[['text', 'amateur_label']]
    df['id'] = range(len(df))
    df = df[['id', 'text', 'amateur_label']]
    df.to_json('huggingface_data.jsonl', orient='records', lines=True)
