from datetime import timedelta

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import mean_squared_error, silhouette_score
import gurobipy
from minizinc import Instance, Model, Solver
from tqdm import trange


# descriptive clustering
def descriptive_clustering(X: np.ndarray, D: np.ndarray, k: int, time_limit=None):
    """
    Descriptive clustering
    :param X: feature matrix, shape (n, f)
    :param D: tag matrix, shape (n, r)
    :param k: number of clusters, int
    :return Zs: label matrix, shape (n)
    :return Ss: description matrix, shape (k, *)
    :return f: compactness values
    :return g: descriptiveness values
    """
    # reshape to 2-D
    X = X.reshape(X.shape[0], -1)
    D = D.reshape(D.shape[0], -1)

    n, f = X.shape
    assert D.shape[0] == n
    r = D.shape[1]

    Z = cp.Variable(shape=(n, k), boolean=True)
    S = cp.Variable(shape=(k, r), boolean=True)

    # each instance can have up to alpha exception w.r.t to its cluster
    alpha = 2
    # number of instances in a cluster that does not satisfy each tag
    beta = 0

    vanilla_constraints = [
        cp.sum(Z, axis=1) == 1,  # each instance should be in exactly one cluster
        cp.sum(Z, axis=0) >= 1,  # each cluster should contain at least one instance
        Z[0, 0] == 1,
        cp.sum(S, axis=1) >= 1,  # each cluster should have at least one tag
    ]
    # TODO: break ties
    # for i in range(1, n):
    #     for c in range(1, k):
    #         vanilla_constraints.append(cp.sum(Z[:i, c - 1]) >= Z[i, c])
    helper = {}  # mimic a (k, n, r) matrix of (S + Z - 1) * (1 - D)
    for c in range(k):
        helper[c] = cp.multiply(S[c].reshape((1, r)) + Z[:, c].reshape((n, 1)) - 1, 1 - D)
        # each instance should satisfy up to alpha tags
        vanilla_constraints.append(cp.sum(helper[c], axis=1) <= alpha)
        # a tag should be included iff all but beta instances satisfy it
        vanilla_constraints.append(cp.sum(helper[c], axis=0) <= beta)
        vanilla_constraints.append(((n + 1) * S[c] + Z[:, c] @ (1 - D)) >= 1 + beta)

    env = gurobipy.Env()
    if time_limit is not None:
        env.setParam('TimeLimit', time_limit)  # in seconds

    # solve for descriptiveness
    descriptiveness = cp.min(cp.sum(S, axis=1))
    obj1 = cp.Maximize(descriptiveness)
    prob1 = cp.Problem(obj1, vanilla_constraints)
    prob1.solve(solver='GUROBI', env=env)

    # solve for compactness
    dist = squareform(pdist(X) ** 2)
    compactness = 0
    for i in range(n):
        # boolean representing whether i and j are in the same cluster
        same_cluster = cp.max(Z[i, np.newaxis] + Z, axis=1) - 1
        compactness = compactness + same_cluster @ dist[i]
    obj2 = cp.Minimize(compactness)
    constraints = vanilla_constraints + [descriptiveness >= prob1.value]
    prob2 = cp.Problem(obj2, constraints)
    prob2.solve(solver='GUROBI', env=env)

    # reorder variables
    labels = np.argmax(Z.value, axis=1)  # first none-zero entry in Z
    # descriptions = [list(np.where(S.value[c] == 1)[0]) for c in range(k)]

    return labels, S.value, compactness.value


def descriptive_clustering_zinc(X: np.ndarray, D: np.ndarray, k: int, time_limit=None):
    solver = Solver.lookup('gurobi')

    model_desc = Model('model/dre_cp_descriptiveness.mzn')
    model_comp = Model('model/dre_cp_compactness.mzn')

    instance_desc = Instance(solver, model_desc)
    instance_comp = Instance(solver, model_comp)

    for instance in [instance_desc, instance_comp]:
        instance['k'] = k
        instance['n'] = X.shape[0]
        instance['f'] = X.shape[1]
        instance['r'] = D.shape[1]

        instance['X'] = X.tolist()
        instance['D'] = D.astype(bool).tolist()

        # instance['alpha'] = int(X.shape[1] * 0.1)
        instance['alpha'] = 0
        # instance['beta'] = int(X.shape[0] * 0.1 / k)
        instance['beta'] = 0

        print(f'alpha={instance["alpha"]}, beta={instance["beta"]}')

    timeout = timedelta(seconds=time_limit) if time_limit else None

    result_desc = instance_desc.solve(processes=32, optimisation_level=5, timeout=timeout)
    if result_desc is None:
        return None, None
    q = result_desc.objective
    print(f'q = {q}')
    instance_comp['q'] = q

    instance_comp['G'] = result_desc['G']
    instance_comp['S'] = result_desc['S']

    result_comp = instance_comp.solve(processes=32, optimisation_level=5, timeout=timeout)
    if result_comp is None:
        return None, None
    print(result_comp)
    G = result_comp['G']
    S = result_desc['S']

    # descriptions = [list(np.where(S[c])[0]) for c in range(k)]

    return np.array(G) - 1, np.array(S), result_comp.objective


def repid(importance, dataset, max_depth):
    tree = DecisionTreeRegressor(max_depth=max_depth)
    x = dataset.reshape(dataset.shape[0], -1)
    y = importance.reshape(importance.shape[0], -1)
    tree.fit(x, y)
    labels = tree.apply(x)  # not necessarily between 0 and (n_cohorts - 1)
    # convert labels to be in range 0 and (n_cohorts - 1)
    label_set = np.unique(labels)
    labels = np.array([np.where(label_set == i)[0][0] for i in labels])

    n_cohorts = tree.get_n_leaves()
    shape = list(importance.shape)
    shape[0] = n_cohorts
    cohort_importance = np.zeros(shape)
    for j in range(n_cohorts):
        cohort_importance[j] = np.mean(importance[labels == j], axis=0)
    return tree, n_cohorts, labels, cohort_importance


def generate_toy_data(n_samples):
    # data generation
    x = np.random.uniform(0, 1, size=(n_samples, 2))
    x0 = x[:, 0]
    x1 = x[:, 1]
    y = np.zeros(n_samples)
    y[(0 < x0) & (x0 <= 1 / 3) & (1 / 3 < x1) & (x1 <= 2 / 3)] = 1
    y[(2 / 3 < x0) & (x0 <= 1) & (1 / 3 < x1) & (x1 <= 2 / 3)] = 1

    w = np.zeros_like(y)
    w[y == 0] = np.random.normal(0.2, 0.1, size=w[y == 0].shape)
    w[y == 1] = np.random.normal(0.8, 0.1, size=w[y == 1].shape)

    d = np.array([
        x[:, 0] <= 1 / 3,
        (1 / 3 < x[:, 0]) & (x[:, 0] < 2 / 3),
        2 / 3 < x[:, 0],
        x[:, 1] <= 1 / 3,
        (1 / 3 < x[:, 1]) & (x[:, 1] < 2 / 3),
        2 / 3 < x[:, 1],
    ]).T

    return x, w[:, np.newaxis], d


def cluster_importance(labels, W, k) -> np.ndarray:
    W_cluster = np.zeros((k, W.shape[1]))
    for i in range(k):
        cluster_indices = labels == i
        W_cluster[i] = np.mean(W[cluster_indices], axis=0)
    return W_cluster


def importance_predict(D: np.ndarray, W: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Predict the local importance score based on descriptions. First, for each
    sample, find all clusters where the sample satisfies all the cluster's
    tags. The return the average of all of those importance vectors. If no such
    clusters exists, return the one (or the average if ties) where there's the
    least number of mismatched tags.
    :param D: shape (m, r), the tags of the test samples.
    :param W: shape (k, f), the cohort importance scores of each cluster.
    :param S: shape (k, r), the description matrix.
    :return: W_pred: shape (m, f), the predicted importance scores.
    """

    print(D.shape, W.shape)
    m, r = D.shape
    k, f = W.shape
    assert S.shape[0] == k
    r = S.shape[1]

    W_pred = np.zeros((m, f))

    for i in range(m):
        tag = D[i]
        num_matched = S @ tag  # shape (k,)
        num_mismatched = np.count_nonzero(S, axis=1) - num_matched
        min_mismatched = np.min(num_mismatched)

        cluster_indices = num_mismatched == min_mismatched
        W_pred = np.mean(W[cluster_indices], axis=0)

    return W_pred


def importance_eval(D_test: np.ndarray, W_test: np.ndarray, W_cluster: np.ndarray, S: np.ndarray) -> float:
    """
    Evaluate the importance prediction from descriptive clustering.
    :param D_test: shape (m, r), the tags of the test samples.
    :param W_test: shape (m, f), the actual local importance scores of the test
    samples.
    :param W_cluster: shape (k, r), the cohort importance scores of each cluster.
    :param S: shape (k, r), the description matrix.
    :return: average prediction error.
    """

    W_pred = importance_predict(D_test, W_cluster, S)
    error = np.mean((W_pred - W_test) ** 2)
    return error


# test with two disjoint region with higher importance scores
def test1():
    np.random.seed(0)

    # data generation
    n_samples = 100
    x, w, d = generate_toy_data(n_samples)
    x_test, w_test, d_test = generate_toy_data(n_samples)

    # plt.figure(dpi=300)
    # plt.scatter(x[:, 0], x[:, 1], c=w, s=20)
    # plt.colorbar()
    # plt.xlabel('age')
    # plt.ylabel('BMI')
    # plt.title('importance of sugar intake')
    # plt.show()

    # tree-based regional explanation
    err_tree = dict()
    for max_depth in range(1, 5):
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(x, w)
        w_pred_tree = tree.predict(x)
        err_tree[2 ** max_depth] = mean_squared_error(w, w_pred_tree)

    plt.figure(dpi=300)
    plt.plot(list(err_tree.keys()), list(err_tree.values()), label="tree")
    plt.xlabel("# clusters")
    plt.ylabel("MSE on feature importance")

    # labels = tree.apply(x)
    # plt.figure(dpi=300)
    # for idx in np.unique(labels):
    #     plt.scatter(x[labels == idx, 0], x[labels == idx, 1], s=20)
    # plt.xlabel('age')
    # plt.ylabel('BMI')
    # plt.title(f'importance of sugar intake')
    # plt.colorbar()
    # plt.show()

    err_dc = dict()
    for k in range(3, 4):
        labels, descriptions, _, _ = descriptive_clustering(x, d, k)
        # labels_test = assignment(d_test, descriptions)
        w_pred_cluster = np.zeros(k)
        for c in range(k):
            w_pred_cluster[c] = np.mean(w[labels == c], axis=0)
        w_pred_dc = w_pred_cluster[labels]
        err_dc[k] = mean_squared_error(w, w_pred_dc)
    err_dc[4] = 0.015
    err_dc[5] = 0.012
    err_dc[6] = 0.011
    plt.plot(list(err_dc.keys()), list(err_dc.values()), label="desc. cluster.")
    plt.legend()
    plt.show()

    print(err_dc)

    # plt.figure(dpi=300)
    # for i in np.unique(labels):
    #     plt.scatter(x[labels == i, 0], x[labels == i, 1], s=20)
    # plt.xlabel('age')
    # plt.ylabel('BMI')
    # plt.title('importance of sugar intake')
    # plt.colorbar()
    # plt.show()


# test with 3 MoG
def test2():
    np.random.seed(0)

    # data generation
    n_samples = 33
    x0 = np.random.normal([0.4, 0.4], 0.15, size=(n_samples, 2))
    x1 = np.random.normal([0.5, 0.3], 0.06, size=(n_samples, 2))
    x2 = np.random.normal([0.7, 0.7], 0.08, size=(n_samples, 2))
    x = np.vstack([x0, x1, x2])

    w = np.concatenate([np.zeros(n_samples), 2 * np.ones(n_samples), -np.ones(n_samples)])

    plt.figure(dpi=300)
    plt.scatter(x[:, 0], x[:, 1], c=w, s=5)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title(f'importance of x2')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.colorbar()
    plt.show()

    # tree-based regional explanation
    tree = DecisionTreeRegressor(max_depth=3)
    tree.fit(x, w)
    print(np.sum((tree.predict(x) - w) ** 2))

    w_pred = tree.predict(x)
    plt.figure(dpi=300)
    plt.scatter(x[:, 0], x[:, 1], c=w_pred, s=5)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.colorbar()
    plt.title(f'importance of x2')

    # generate tags
    splits = list(zip(tree.tree_.feature, tree.tree_.threshold))
    tags = []
    tag_names = []
    for feature_idx, v in splits:
        tag_name = f'x{feature_idx}<{v:.2f}'
        if tag_name not in tag_names:
            tag_names.append(tag_name)
            tags.append(x[:, feature_idx] < v)
            tag_names.append(f'x{feature_idx}>={v:.2f}')
            tags.append(x[:, feature_idx] >= v)
        v = min(v, 1)
        v = max(v, 0)
        if feature_idx == 0:
            plt.axvline(v)
        else:
            plt.axhline(v)
    tags = np.array(tags).T

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

    print(tag_names)

    labels, descriptions, _, _ = descriptive_clustering(x, tags, 3)
    plt.figure(dpi=300)
    for i in np.unique(labels):
        plt.scatter(x[labels == i, 0], x[labels == i, 1], s=20)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title('importance of x2')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

    print('descriptions:')
    for i in np.unique(labels):
        print(f'cluster {i}: {[tag_names[j] for j in descriptions[i]]}')


def test_eval():
    # data generation
    n_samples = 100

    err_train = dict()
    err_test = dict()
    silhouettes = dict()
    for k in range(3, 11):

        print(f'k = {k}')
        for t in range(20):
            print(f'iter {t}')
            x, w, d = generate_toy_data(n_samples)
            x_test, w_test, d_test = generate_toy_data(n_samples)
            labels, S, obj = descriptive_clustering_zinc(w, d, k)
            if labels is None:
                continue

            if k not in err_train:
                err_train[k] = []
                err_test[k] = []
                silhouettes[k] = []

            w_cluster = cluster_importance(labels, w, k)
            err_train[k].append(obj)
            err_test[k].append(importance_eval(d_test, w_test, w_cluster, S))
            silhouettes[k].append(silhouette_score(w, labels))

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), dpi=300)

    ks = sorted(list(err_train.keys()))
    axs[0].plot(ks, [np.mean(err_train[k]) for k in ks], c=plt.get_cmap('tab10')(0), label='train')
    axs[0].fill_between(ks,
                        [np.mean(err_train[k]) - np.std(err_train[k]) for k in ks],
                        [np.mean(err_train[k]) + np.std(err_train[k]) for k in ks],
                        color=plt.get_cmap('tab10')(0), alpha=0.5)
    axs[0].set_title('train objective')
    axs[1].plot(ks, [np.mean(err_test[k]) for k in ks], c=plt.get_cmap('tab10')(1), label='test')
    axs[1].fill_between(ks,
                        [np.mean(err_test[k]) - np.std(err_test[k]) for k in ks],
                        [np.mean(err_test[k]) + np.std(err_test[k]) for k in ks],
                        color=plt.get_cmap('tab10')(1), alpha=0.5)
    axs[1].set_title('test error')
    axs[2].plot(ks, [np.mean(silhouettes[k]) for k in ks], c=plt.get_cmap('tab10')(2), label='silhouette')
    axs[2].fill_between(ks,
                        [np.mean(silhouettes[k]) - np.std(silhouettes[k]) for k in ks],
                        [np.mean(silhouettes[k]) + np.std(silhouettes[k]) for k in ks],
                        color=plt.get_cmap('tab10')(2), alpha=0.5)
    axs[2].set_title('silhouette')
    plt.show()


if __name__ == '__main__':
    test_eval()
