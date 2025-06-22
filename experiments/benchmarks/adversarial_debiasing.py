import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)


class Classifier(nn.Module):
    def __init__(self, hidden_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.fc(x)


class AdversarialClassifier(nn.Module):
    def __init__(self, hidden_dim):
        super(AdversarialClassifier, self).__init__()
        self.fc = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        return self.fc(x)


def load_and_preprocess_data(data_dir):
    df_train = pd.read_json(os.path.join(data_dir, 'train.jsonl'), lines=True)
    X_train = df_train['processed_text']
    y_train = df_train['expert_label']
    z_train = df_train['group'].map({'aae': 0, 'white': 1, 'others': 2})
    return X_train, y_train, z_train


def vectorize_text(X_train):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_train_tensor = torch.tensor(X_train_vec, dtype=torch.float32)
    return vectorizer, X_train_tensor


def create_model(input_dim, hidden_dim, learning_rate):
    encoder = Encoder(input_dim, hidden_dim)
    classifier = Classifier(hidden_dim)
    adversary = AdversarialClassifier(hidden_dim)

    optimizer_enc_cls = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=learning_rate)
    optimizer_adv = optim.Adam(adversary.parameters(), lr=learning_rate)

    return encoder, classifier, adversary, optimizer_enc_cls, optimizer_adv


def pretrain_encoder_classifier(encoder, classifier, optimizer, X_train_tensor, y_train_tensor, epochs):
    criterion_cls = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        encoded = encoder(X_train_tensor)
        predictions = classifier(encoded)
        loss = criterion_cls(predictions, y_train_tensor)
        loss.backward()
        optimizer.step()


def adversarial_training(encoder, classifier, adversary, optimizer_enc_cls, optimizer_adv, X_train_tensor, y_train_tensor, z_train_tensor, alpha, epochs):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        optimizer_adv.zero_grad()
        with torch.no_grad():
            encoded = encoder(X_train_tensor)
        adv_predictions = adversary(encoded.detach())
        loss_adv = criterion_adv(adv_predictions, z_train_tensor)
        loss_adv.backward()
        optimizer_adv.step()

        optimizer_enc_cls.zero_grad()
        encoded = encoder(X_train_tensor)
        cls_predictions = classifier(encoded)
        adv_predictions = adversary(encoded)

        loss_cls = criterion_cls(cls_predictions, y_train_tensor)

        uniform_target = torch.full_like(adv_predictions, 1 / 3)
        loss_enc_adv = -(nn.functional.log_softmax(adv_predictions, dim=1) * uniform_target).sum(dim=1).mean()

        loss_total = alpha * loss_cls + (1 - alpha) * loss_enc_adv
        loss_total.backward()
        optimizer_enc_cls.step()


def main(args):
    X_train, y_train, z_train = load_and_preprocess_data(args.data_dir)
    vectorizer, X_train_tensor = vectorize_text(X_train)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    z_train_tensor = torch.tensor(z_train.values, dtype=torch.long)

    encoder, classifier, adversary, optimizer_enc_cls, optimizer_adv = create_model(
        X_train_tensor.shape[1], args.hidden_dim, args.lr
    )

    pretrain_encoder_classifier(encoder, classifier, optimizer_enc_cls, X_train_tensor, y_train_tensor, args.epochs_pretrain)
    adversarial_training(encoder, classifier, adversary, optimizer_enc_cls, optimizer_adv,
                         X_train_tensor, y_train_tensor, z_train_tensor, args.alpha, args.epochs_adv)

    encoder.eval()
    classifier.eval()

    df_test = pd.read_json(os.path.join(args.data_dir, 'test.jsonl'), lines=True)
    all_results = []
    for text in df_test['processed_text'].to_list():
        try:
            test_vec = vectorizer.transform([text]).toarray()
            test_tensor = torch.tensor(test_vec, dtype=torch.float32)

            with torch.no_grad():
                encoded_test = encoder(test_tensor)
                logits = classifier(encoded_test)
                label = torch.argmax(logits, dim=1).item()
                all_results.append(label)
        except:
            print('Error: ', text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        type=str, 
                        default='', 
                        help='Directory to load the data'
                        )
    parser.add_argument('--hidden_dim', 
                        type=int, 
                        default=300, 
                        help='Dimension of the hidden layer'
                        )
    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.05, 
                        help='Weight for the adversarial loss'
                        )
    parser.add_argument('--epochs_pretrain', 
                        type=int, 
                        default=15, 
                        help='Number of pretraining epochs'
                        )
    parser.add_argument('--epochs_adv', 
                        type=int, 
                        default=15, 
                        help='Number of adversarial training epochs'
                        )
    parser.add_argument('--lr', 
                        type=float, 
                        default=5e-5, 
                        help='Learning rate'
                        )

    args = parser.parse_args()
    main(args)