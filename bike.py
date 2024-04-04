import os

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

from explainer import SHAPExplainer
from bike_sharing_scenario import BikeScenario

from descriptive_cluster import descriptive_clustering, repid

n_samples = 50

# set seed
np.random.seed(0)

# load data
scenario = BikeScenario()
x = scenario.X[:n_samples]
y = scenario.y[:n_samples]

importance_file = 'data/bike/importance_bike1000.npy'
if not os.path.exists(importance_file):
    explainer = SHAPExplainer(scenario.model.predict, mode='default')
    importance = explainer.explain(x)
    print(importance.shape)
    np.save(importance_file, importance)
else:
    importance = np.load(importance_file)

importance = importance[:n_samples]

k_repid = []
errors_repid = []
for max_depth in range(1, 6):
    tree, k, labels, cohort_importance = repid(importance, x, max_depth=max_depth)
    error = 0
    for j in range(k):
        importance_diff = (importance[labels == j] - cohort_importance[j]) ** 2  # shape (n_j, num_features)
        error += np.sum(np.mean(importance_diff, axis=1))
    k_repid.append(k)
    errors_repid.append(error / x.shape[0])
plt.figure(dpi=300)
plt.plot(k_repid, errors_repid, label='REPID')
plt.xlabel('# cohorts')
plt.ylabel('compactness')
plt.title('REPID: compactness vs. conciseness')

tree, k, labels, cohort_importance = repid(importance, x, max_depth=3)
x_df = pd.DataFrame({f: arr for f, arr in zip(scenario.feature_names, x.T)})
# generate tags
splits = list(zip(x_df.columns[tree.tree_.feature], tree.tree_.threshold))
tags = []
tag_names = []
for f, v in splits:
    tag_name = f'{f}<{v:.2f}'
    if tag_name not in tag_names:
        tag_names.append(tag_name)
        tags.append(x_df[f] < v)
        tag_names.append(f'{f}>={v:.2f}')
        tags.append(x_df[f] >= v)
tags = np.array(tags).T

errors_desc = []
for k in range(2, 10):
    labels, descriptions = descriptive_clustering(x, tags, k=k)
    error = 0
    for j in range(k):
        _cohort_importance = np.mean(importance[labels == j], axis=0)
        importance_diff = (importance[labels == j] - _cohort_importance) ** 2  # shape (n_j, num_features)
        error += np.sum(np.mean(importance_diff, axis=1))
    errors_desc.append(error / x.shape[0])
plt.plot(range(2, 10), errors_desc, label='tag-based')
plt.legend()
plt.show()

################
# eval importance on hour, partition on temp and wind_speed
################
# x = pd.DataFrame({f: x[:, i] for i, f in enumerate(scenario.feature_names)})
#
# print('feature with highest importance:')
# print(x.columns[np.argmax(np.mean(np.abs(importance), axis=0))])
# feature_idx = np.argmax(np.mean(np.abs(importance), axis=0))
#
# plt.figure(dpi=300)
# plt.scatter(x['temp'].values[:n_samples], x['wind_speed'].values[:n_samples],
#             c=importance[:n_samples, feature_idx], s=20)
# plt.xlabel('temp')
# plt.ylabel('wind_speed')
# plt.title(f'importance of {x.columns[feature_idx]}')
# plt.colorbar()
# plt.show()
#
# # REPID
# tree = DecisionTreeRegressor(max_depth=3)
# x_tree = x[['temp', 'wind_speed']].loc[:n_samples]
# tree.fit(x[['temp', 'wind_speed']], importance)
# labels = tree.apply(x_tree)
# plt.figure(dpi=300)
# for idx in np.unique(labels):
#     plt.scatter(x_tree.loc[labels == idx, 'temp'], x_tree.loc[labels == idx, 'wind_speed'], s=20)
# plt.xlabel('temp')
# plt.ylabel('wind_speed')
# plt.title(f'importance of {x.columns[feature_idx]}')
# plt.colorbar()
# plt.show()
#
# # generate tags
# splits = list(zip(x_tree.columns[tree.tree_.feature], tree.tree_.threshold))
# tags = []
# tag_names = []
# for f, v in splits:
#     tag_name = f'{f}<{v:.2f}'
#     if tag_name not in tag_names:
#         tag_names.append(tag_name)
#         tags.append(x_tree[f] < v)
#         tag_names.append(f'{f}>={v:.2f}')
#         tags.append(x_tree[f] >= v)
# print(tag_names)
# tags = np.array(tags).T
#
# # descriptive clustering
# print(x_tree.values.shape, tags.shape)
# labels, descriptions = descriptive_clustering(x_tree.values, tags, 4)
# for i in range(4):
#     print(f'cluster {i}: description={[tag_names[j] for j in descriptions[i]]}')
# plt.figure(dpi=300)
# for i in np.unique(labels):
#     plt.scatter(x_tree['temp'].loc[labels == i], x_tree['wind_speed'].loc[labels == i], s=20)
# plt.xlabel('temp')
# plt.ylabel('windspeed')
# plt.title(f'importance of {x.columns[feature_idx]}')
# plt.colorbar()
# plt.show()
