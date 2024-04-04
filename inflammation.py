#!/usr/bin/python3
from datetime import datetime

import os
import warnings

import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

from descriptive_cluster import descriptive_clustering, descriptive_clustering_zinc, repid, importance_eval, \
    cluster_importance
from explainer import LIMEExplainer, SHAPExplainer

startTime = datetime.now()

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# set working directory
workdir = os.getcwd()
print('Working directory is {}'.format(workdir))

# set seed
np.random.seed(0)

# load data
X_train = pd.read_csv("data/inflammation/x_train_taxahfe_no_sf.csv", index_col=0, header=0)
y_train = pd.read_csv("data/inflammation/y_train_class.csv", index_col=0, header=0)
sw_train = pd.read_csv("data/inflammation/sw_train_taxahfe.csv", index_col=0, header=0)
x_test = pd.read_csv('data/inflammation/x_test_no_sf.csv', index_col=0)
y_test = pd.read_csv('data/inflammation/y_test_class.csv', index_col=0)
x_test = x_test[X_train.columns]  # reordered columns to match (column ordered changed during taxahfe)

X_train = X_train.sort_index(axis=0)
y_train = y_train.sort_index(axis=0)
sw_train = sw_train.sort_index(axis=0)
x_test = x_test.sort_index(axis=0)
y_test = y_test.sort_index(axis=0)

# model
params = {'subsample': 0.7, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.05,
          'gamma': 0.4, 'colsample_bytree': 0.5}

xgb = xgb.XGBClassifier(**params)
# xgb.fit(X_train, y_train.values.ravel(), sample_weight=sw_train.values.ravel())

# y_pred = xgb.predict(x_test)
# print("Test Acc: {:.3f}".format(accuracy_score(y_test, y_pred)))

cont_features = ['Age', 'family_pir', 'BMI']
discrete_features = np.array(['Sex_Female', 'Sex_Male', 'education_college graduate',
                              'education_high school graduate or equivalent',
                              'education_less than high school graduate', 'education_some college',
                              'Ethnicity_Mexican_American', 'Ethnicity_Non-Hispanic_Black',
                              'Ethnicity_Non-Hispanic_White', 'Ethnicity_Other_Hispanic',
                              'Ethnicity_Other_Multi-Racial', 'ever_smoker_no', 'ever_smoker_yes',
                              'diabetes_no', 'diabetes_yes', 'hypertension_no', 'hypertension_yes'])
food_features_idx = [i for i in range(len(x_test.columns)) if x_test.columns[i] not in discrete_features
                     and x_test.columns[i] not in cont_features]

# importance
importance_file = 'data/inflammation/importance_shap.npy'
if not os.path.exists(importance_file):
    explainer = SHAPExplainer(xgb, mode='default')
    importance = explainer.explain(x_test.values)
    print(importance.shape)
    np.save(importance_file, importance)
else:
    importance = np.load(importance_file)

importance = importance[:, food_features_idx]

# running descriptive clustering
num_trials = 5
ks = np.arange(3, 11, dtype=int)
objs = np.zeros((np.max(ks) + 1, num_trials))
errs = np.zeros((np.max(ks) + 1, num_trials))
silhouette = np.zeros((np.max(ks) + 1, num_trials))
kf = KFold(n_splits=num_trials, shuffle=True, random_state=42)
splits = kf.split(x_test)
for k in ks:
    for t, (optim_idx, valid_idx) in enumerate(splits):
        print(f'======== k = {k}, iter {t}/{num_trials} ========')
        try:
            labels, S, obj = descriptive_clustering_zinc(importance[optim_idx],
                                                         x_test[discrete_features].values[optim_idx],
                                                         k, 300)
        except Exception:
            continue

        W_cluster = cluster_importance(labels, importance[optim_idx], k)

        objs[k, t] = obj
        errs[k, t] = importance_eval(x_test[discrete_features].values[valid_idx],
                                     importance[valid_idx], W_cluster, S)
        silhouette[k, t] = silhouette_score(importance[optim_idx], labels)

        np.save(f'./output/shap_all/labels_k={k}_t={t}.npy', labels)
        np.save(f'./output/shap_all/descriptions_k={k}_t={t}.npy', S)
        np.save(f'./output/shap_all/importance_k={k}_t={t}.npy', W_cluster)
        np.save(f'./output/shap_all/idx_test_k={k}_t={t}.npy', optim_idx)
        np.save(f'./output/shap_all/objs.npy', objs)
        np.save(f'./output/shap_all/errs.npy', errs)

print(f'err = {errs}')
print(f'silhouette = {silhouette}')
