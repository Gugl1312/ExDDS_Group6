import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


def import_dataset(n):
    if n == 1:  # Netflix dataset
        data = pd.read_csv('data/Netflix/data_final.csv')
        return data
    elif n == 2:  # Jester dataset
        X = pd.read_csv('data/Jester/jester_final.csv')
        X.columns = X.columns.astype(str)
        X.replace(999, np.nan, inplace=True)
        X.rename(columns={'0': 'user_id'}, inplace=True)
        X['user_id'] = range(1, len(X) + 1)
        return X
    elif n == 3:  # Goodreads10K dataset
        data = pd.read_csv('data/Goodreads10K/ratings_final.csv')
        return data
    else:
        print("Error: invalid dataset number")
        return None


def clusterUsers(X, n_clusters, max_iter=10):
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


def Ggh(g, h, i, mean, var):  # Г_gh(v)
    return ((mean.loc[g][i] + mean.loc[h][i]) ** 2) / (var.loc[g][i])


def calc_alpha(g, h, i, mean, var, V_n):
    total = 0
    for v in V_n:
        a = v["entity_id"]
        total += Ggh(g, h, a, mean, var)
    return Ggh(g, h, i, mean, var) / total


def get_performance(user_ratings, initial_cluster, est_cluster):
    # Accuracy
    accuracy = (initial_cluster == est_cluster)

    # Regret
    user_ratings = pd.DataFrame(user_ratings)
    user_ratings['rating'].astype(float)
    user_ratings_sorted = sorted(user_ratings.values, key=lambda x: x[1], reverse=True)
    # print('sui', user_ratings_sorted[0][1])
    highest_rating = user_ratings_sorted[0][1]

    regret = 0
    for i in range(len(user_ratings_sorted)):
        generated_rating = user_ratings_sorted[i][1]
        regret += float(highest_rating) - float(generated_rating)

    # Convergence time
    final_value = round(0.8 * regret)
    convergence_time = 0
    regr = 0

    for i in range(len(user_ratings.values)):
        convergence_time += 1
        generated_rating = user_ratings.values[i][1]
        regr += float(highest_rating) - float(generated_rating)

        if regr >= final_value:
            break

    return accuracy, regret, convergence_time


def generateRatings(mean, var, g, v):
    return np.random.normal(mean.loc[g][v], var.loc[g][v])


def g_hat_select(V_n, G, mean, var):
    g_hat = None
    h_hat_max = None
    maxVal = None
    for g in G["cluster"].unique():
        minVal = None
        h_hat_min = None
        for h in G["cluster"].unique():
            if g != h:
                Rn = abs(calc_Rn(V_n, g, h, mean, var))
                if minVal is None or Rn < minVal:
                    minVal = Rn
                    h_hat_min = h
        if maxVal is None or minVal > maxVal:
            g_hat = g
            h_hat_max = h_hat_min
            maxVal = minVal
    return g_hat, h_hat_max


def calc_Rn(V_n, g, h, mean, var):
    total = 0
    index = 0
    for v in V_n:
        i = v["entity_id"]
        total += calc_alpha(g, h, i, mean, var, V_n) * (V_n[index]["rating"] - mean.loc[h][i]) / (
                mean.loc[g][i] - mean.loc[h][i])
        index += 1
    return total


def define_candidate_set(V_n, G, mean, var, C, n):
    M_n = []
    for g in G["cluster"].unique():
        minVal = None
        for h in G["cluster"].unique():
            if g != h:
                Rn = abs(calc_Rn(V_n, g, h, mean, var))
                if minVal is None or Rn < minVal:
                    minVal = Rn

        if abs(minVal - 1) <= C:
            M_n.append(g)
    return M_n


def g_hat_select2(V_n, G, mean, var, n):
    g_hat = None
    minVal = None
    for g in G["cluster"].unique():
        for h in G["cluster"].unique():
            if g != h:
                total = 0
                for v in V_n:
                    i = v["entity_id"]
                    total += Ggh(g, h, i, mean, var)
                if minVal is None or total < minVal:
                    minVal = total
                    g_hat = g
    return g_hat


def calc_sigma_n(g, h, V_n, mean, var):
    total = 0
    for v in V_n:
        i = v["entity_id"]
        total += Ggh(g, h, i, mean, var)
    return total


def clusterBasedBanditAlgorithm(B, C, D, G, mean, var, g_actual):  # Algorithm 1
    V_n = []

    g_hat = random.randint(0, len(G['cluster'].unique()) - 1)
    V_n = g_exploration(V_n, g_hat, G, mean, var, g_actual)
    for i in range(25):  # Algorithm should converge within 25 iterations
        M_n = define_candidate_set(V_n, G, mean, var, C, i)
        if len(M_n) > 0:
            g_hat, h = g_hat_select(V_n, G, mean, var)
            V_n = g_exploration(V_n, g_hat, G, mean, var, g_actual)
            if calc_sigma_n(g_hat, h, V_n, mean, var) >= B or (
                    len(M_n) == 1 and (i + 1) > D * np.log2(len(G['cluster'].unique()))):
                return V_n, g_hat
        else:
            g_hat = g_hat_select2(V_n, G, mean, var, i)
            V_n = g_exploration(V_n, g_hat, G, mean, var, g_actual)
        if i == 25:
            print("Failed to converge")
    return V_n, g_hat


def g_exploration(V_n, g_hat, G, mean, var, g_actual):  # Algorithm 2
    # print('V_n:', V_n)
    n = len(V_n)
    groups = G['cluster'].unique()

    if n == 0:  # rating the first item
        while True:
            h = random.randint(0, len(G['cluster'].unique()) - 1)
            if h != g_hat:
                break
        max_i = 0
        max_ = -1
        for v in V_n:
            e = v["entity_id"]
            val = Ggh(g_hat, h, e, mean, var)
            if val > max_:
                max_ = val
                max_i = e
        V_n.append({'entity_id': max_i, 'rating': generateRatings(mean, var, g_actual, max_i)})
        # print('returning V_n:', V_n)
        return V_n
    h = -1
    Min_ = np.inf
    for h_i in groups:
        min_ = 0
        for i in range(len(V_n)):
            min_ += Ggh(g_hat, h_i, V_n[i]['entity_id'], mean, var)
        if min_ < Min_:
            Min_ = min_
            h = h_i
    max_i = 0
    max_ = -1
    for v in V_n:
        e = v["entity_id"]
        val = Ggh(g_hat, h, e, mean, var)
        if val > max_:
            max_ = val
            max_i = e
    V_n.append({'entity_id': h, 'rating': generateRatings(mean, var, g_actual, max_i)})
    # print('returning V_n:', V_n)
    return V_n


def evaluate(dataset_n, nclusters):
    print("Evaluating dataset " + str(dataset_n))
    data = import_dataset(dataset_n)
    clusters = clusterUsers(data, nclusters)
    data['cluster'] = clusters
    # Group clusters to calculate mean and variance
    cluster_mean = data.groupby('cluster').mean(numeric_only=True)
    # the fist column contains the information of the user_id which we do not need
    cluster_mean = cluster_mean.drop(["user_id"], axis=1)
    cluster_var = data.groupby('cluster').var(numeric_only=True)
    # the fist column contains the information of the user_id which we do not need
    cluster_var = cluster_var.drop(["user_id"], axis=1)
    G = pd.DataFrame({'user_id': data['user_id'], 'cluster': data['cluster']})

    overall_accuracy = []
    overall_regret = []
    overall_convergence_time = []

    for i in range(4):
        for u in tqdm(range(10), desc="Running the algorithm for different users", total=10):
            user_ratings, g_hat = clusterBasedBanditAlgorithm(5, 0.5, 3, G, cluster_mean, cluster_var, g_actual=i)
            accuracy, regret, convergence_time = get_performance(user_ratings, initial_cluster=i, est_cluster=g_hat)
            overall_accuracy.append(accuracy)
            overall_regret.append(regret)
            overall_convergence_time.append(convergence_time)
    print("Average accuracy: " + str(np.mean(overall_accuracy)))
    print("Average regret: " + str(np.mean(overall_regret)) + "std for regret:" + str(np.std(overall_regret)))
    print("Average convergence time: " + str(np.mean(overall_convergence_time)) + "std for convergence time:" + str(
        np.std(overall_convergence_time)))


def test_all_datasets():
    for i in range(1, 4):  # example to test all of the datasets
        evaluate(i, 4)


def main():
    evaluate(2, 4)


if __name__ == '__main__':
    main()
