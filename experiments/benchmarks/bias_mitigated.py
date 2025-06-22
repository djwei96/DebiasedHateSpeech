import os
import spacy
import argparse
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import wordnet as wn

# Uncomment the following lines to download the required NLTK resources
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('en_core_web_sm')


def detect_bsws(clf, vectorizer, threshold=0.8):
    bsws = []
    for word in vectorizer.get_feature_names_out():
        prob = clf.predict_proba(vectorizer.transform([word]))[0][1]
        if prob >= threshold:
            bsws.append(word.lower())
    return bsws


def replace_pos_tags(text, bsws):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return ' '.join(f"<{token.pos_}>" if token.text.lower() in bsws else token.text for token in doc)


def get_hypernym(word):
    synsets = wn.synsets(word)
    if synsets:
        hypernyms = synsets[0].hypernyms()
        if hypernyms:
            return hypernyms[0].lemmas()[0].name()
    return word


def replace_hypernyms(text, bsws):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return ' '.join(get_hypernym(token.text.lower()) if token.text.lower() in bsws else token.text for token in doc)


def main(args):
    df_train = pd.read_json(os.path.join(args.data_dir, 'train.jsonl'), lines=True)
    df_test = pd.read_json(os.path.join(args.data_dir, 'test.jsonl'), lines=True)
    
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(df_train['processed_text'])
    clf = LogisticRegression().fit(X_train, df_train['expert_label'])
    bsws = detect_bsws(clf, vectorizer)
    
    df_train['text_pos'] = df_train['processed_text'].apply(lambda x: replace_pos_tags(x, bsws))
    df_train['text_hypernym'] = df_train['processed_text'].apply(lambda x: replace_hypernyms(x, bsws))
    
    X_train_pos = vectorizer.fit_transform(df_train['text_pos'])
    clf_pos = LogisticRegression().fit(X_train_pos, df_train['expert_label'])
    
    df_test['text_pos'] = df_test['processed_text'].apply(lambda x: replace_pos_tags(x, bsws))
    X_test_pos = vectorizer.transform(df_test['text_pos'])
    all_results = clf_pos.predict(X_test_pos)
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='', 
                        help='Directory to load the data'
                        )

    args = parser.parse_args()
    main(args)
    