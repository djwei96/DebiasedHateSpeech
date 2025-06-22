import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eig


def refine_label(df,  k_neighbors, label_column='label', eps=1e-5):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text']).toarray()
    y = df[label_column].values
    n_samples = X.shape[0]

    knn_graph = kneighbors_graph(X, n_neighbors=k_neighbors, mode='connectivity', include_self=True).toarray()
    distances = euclidean_distances(X, X)
    sigma = np.mean(np.sort(distances, axis=1)[:, 1:k_neighbors+1], axis=1) + eps
    sigma_matrix = np.outer(sigma, sigma)
    W = np.exp(-distances**2 / (2 * sigma_matrix)) * knn_graph
    np.fill_diagonal(W, W.diagonal() + eps)

    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + eps))
    L_bar = np.eye(n_samples) - D_inv_sqrt @ W @ D_inv_sqrt

    A = (y[:, None] == y[None, :]).astype(float)
    DA = np.diag(A.sum(axis=1))
    DA_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(DA) + eps))
    A_bar = DA_inv_sqrt @ A @ DA_inv_sqrt

    A_bar += np.eye(n_samples) * eps

    eigvals, eigvecs = eig(L_bar, A_bar)
    eigvals = eigvals.real
    eigvecs = eigvecs.real
    idx = eigvals.argsort()
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    f_star = eigvecs[:, 1]

    refined_labels = (f_star > np.median(f_star)).astype(int)
    df['refined_label'] = refined_labels
    return df


def spectral_cluster(data_path, save_dir, k_neighbors=10):
    df_refined = None
    df = pd.read_json(data_path, lines=True)
    df = df.sample(frac=1, random_state=42)
    df_train = df.sample(n=256)
    df_test = df.drop(df_train.index)
    df_test = df_test.sample(frac=0.8)
    df_train['label'] = df_train['expert_label']
    df_test['label'] = df_test['amateur_label']
    df = pd.concat([df_train, df_test], axis=0)

    df_refined = refine_label(df, k_neighbors)
    df_refined = df_refined[['id', 'text', 'refined_label']]
    df_refined.to_json(os.path.join(save_dir, 'spectral_clustering_data.jsonl'), lines=True, orient='records')
    return None


def convert_data_for_confident_learning(data_path, save_dir):
    df = pd.read_json(data_path, lines=True)
    df['text'] = df['text'].apply(lambda x: ' '.join(x.strip().split()))
    df_train = df.sample(n=256)
    df_test = df.drop(df_train.index)
    df_test = df_test.sample(frac=0.8)

    df_train['label'] = df_train['expert_label']
    df_test['label'] = df_test['amateur_label']
    df_train = df_train[['text', 'label']]
    df_test = df_test[['text', 'label']]
    df = pd.concat([df_train, df_test], axis=0)

    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'confident_learning_data.csv'), sep='\t', index=False)
    return None


def convert_data_for_noise_modeling(data_path, save_dir):
    df = pd.read_json(data_path, lines=True)
    df['text'] = df['text'].apply(lambda x: ' '.join(x.strip().split()))

    df_train = df.sample(n=256)
    df_test = df.drop(df_train.index)
    df_val = df_train.sample(frac=0.2)
    df_train = df_train.drop(df_val.index)

    df_train = df_train[['text', 'expert_label']]
    df_val = df_val[['text', 'expert_label']]
    df_test = df_test[['text', 'amateur_label']]
    df_test = df_test.sample(frac=0.8)

    df_train.rename(columns={'expert_label': 'label'}, inplace=True)
    df_val.rename(columns={'expert_label': 'label'}, inplace=True)
    df_test.rename(columns={'amateur_label': 'label'}, inplace=True)

    os.makedirs(save_dir, exist_ok=True)
    df_train.to_csv(os.path.join(save_dir, 'noise_modeling_data_train.tsv'), sep='\t', index=False, header=False)
    df_val.to_csv(os.path.join(save_dir, 'noise_modeling_data_dev.tsv'), sep='\t', index=False, header=False)
    df_test.to_csv(os.path.join(save_dir, 'noise_modeling_data_test.tsv'), sep='\t', index=False, header=False)
    return None


if __name__ == '__main__':
    data_path = 'YOUR_DATA_PATH'
    save_dir = 'YOUR_SAVE_DIR'
    spectral_cluster(data_path, save_dir)
    convert_data_for_noise_modeling(data_path, save_dir)
    convert_data_for_confident_learning(data_path, save_dir)
    