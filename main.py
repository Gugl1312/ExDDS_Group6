import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import warnings
from tqdm import tqdm
import sys
warnings.filterwarnings("ignore")

def import_dataset(n):
    if n == 1: # Netflix dataset
        data = pd.read_csv('data/Netflix/data_final.csv')
        return data
    elif n == 2: # Jester dataset
        X = pd.read_csv('data/Jester/jester_final.csv')
        X.columns = X.columns.astype(str)
        X.replace(999, np.nan, inplace=True)
        X.rename(columns={'0': 'user_id'}, inplace=True)
        X['user_id'] = range(1, len(X) + 1)
        return X
    elif n == 3: # Goodreads10K dataset
        data = pd.read_csv('data/Goodreads10K/ratings_final.csv')
        return data
    else:
        print("Error: invalid dataset number")
        return None

def clusterUsers(X, n_clusters, max_iter=10): # TODO: BLC matrix-factorization clustering algorithm later?
    """Perform K-Means clustering on data with missing values.
       Implementation of the algorithm borrowed from:https://stackoverflow.com/questions/35611465/python-scikit-learn-clustering-with-missing-data
       Based on a paper: https://arxiv.org/pdf/1411.7013.pdf

    Args:
      X: An [n_samples, n_features] array of data to cluster.
      n_clusters: Number of clusters to form.
      max_iter: Maximum number of EM iterations to perform.

    Returns:
      labels: An [n_samples] vector of integer labels.
      centroids: An [n_clusters, n_features] array of cluster centroids.
      X_hat: Copy of X with the missing values filled in.
    """
    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    for i in range(max_iter):
        if i > 0:
            # initialize KMeans with the previous set of centroids. this is much
            # faster and makes it easier to check convergence (since labels
            # won't be permuted on every iteration), but might be more prone to
            # getting stuck in local minima.
            cls = KMeans(n_clusters, init=prev_centroids)
        else:
            # do multiple random initializations in parallel
            cls = KMeans(n_clusters)

        # perform clustering on the filled-in data
        labels = cls.fit_predict(X_hat)
        centroids = cls.cluster_centers_

        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels
        prev_centroids = cls.cluster_centers_

    return labels

def generateRatings(mean, var):
    return np.random.normal(mean, var)

def g_hat_select():
    return

def clusterBasedBanditAlgorithm(B, C, D, G, mean, var): # Algorithm 1
    V_n = []
    print('unique clusters', len(G['cluster'].unique()))
    g_hat = G.iloc[random.randint(0, len(G['cluster'].unique()) - 1)]

    Mn = ...

    if Mn is not None:
        g_hat_select()
        g_exploration()
    n = 0
    while n < 25:
        candidate = ...
        if candidate is not None:
            ...
    return


def g_exploration(V_n, g_hat, mean, var, n): # Algorithm 2
    n = len(V_n) + 1
    for i in range(n):
        pass
    return

def Ggh(g, h, v, mean, var):
    return ((mean[mean['cluster'] == g][str(g)] + mean[mean['cluster'] == g][str(h)])**2) / (var[var['cluster' == g]][str(g)])

def evaluate(dataset_n):
    print("Evaluating dataset " + str(dataset_n))
    data = import_dataset(dataset_n)
    print(data)
    for i in [4, 8, 16, 32]:
        print("Running algorithm with " + str(i) + " clusters")
        clusters = clusterUsers(data, i)
        print(clusters)
        data['cluster'] = clusters
        cluster_mean = data.groupby('cluster').mean(numeric_only=True)
        cluster_var = data.groupby('cluster').var(numeric_only=True)

        G = pd.DataFrame({'user_id': data['user_id'], 'cluster': data['cluster']})
        clusterBasedBanditAlgorithm(5, 0.5, 3, G, cluster_mean, cluster_var)    
        return

def test_all_datasets():
    for i in range(1, 4): # example to test all of the datasets
        evaluate(i)

def main():
    evaluate(2)


if __name__ == '__main__':
    main()