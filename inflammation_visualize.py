import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb

from descriptive_cluster import importance_eval

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

# ======== model ========
params = {'subsample': 0.7, 'n_estimators': 200, 'min_child_weight': 3, 'max_depth': 3, 'learning_rate': 0.05,
          'gamma': 0.4, 'colsample_bytree': 0.5}
xgb = xgb.XGBClassifier(**params)
xgb.fit(X_train, y_train.values.ravel(), sample_weight=sw_train.values.ravel())
y_pred: np.ndarray = xgb.predict(x_test)
correct: np.ndarray = y_test.values.flatten() == y_pred

# ======== global and local importance =========
for name in ['lime', 'shap_tree']:
    importance = np.load(f'data/inflammation/importance_{name}.npy')
    # plot local importance
    plt.figure(dpi=300, figsize=(10, 10))
    df = pd.DataFrame(importance, columns=x_test.columns)
    sns.stripplot(df, s=1, orient='h', alpha=0.2)
    plt.tight_layout()
    plt.show()

    # plot global importance
    plt.figure(dpi=300, figsize=(10, 10))
    sns.barplot(df, orient='h')
    plt.tight_layout()
    plt.show()

# ======== objs and errs ========
objs = np.load('./output/shap_all/objs.npy')
errs = np.load('./output/shap_all/errs.npy')

for i, (vals, label) in enumerate(zip([objs, errs], ['optim. obj', 'valid. err'])):
    plt.figure(dpi=300)
    xs = range(3, 6)
    mean = np.mean(vals, axis=1)
    std = np.mean(vals, axis=1)
    # plt.plot(xs, mean[xs], label=label, color=plt.get_cmap('tab10')(i))
    # plt.fill_between(xs, mean[xs] - std[xs], mean[xs] + std[xs], alpha=0.5, color=plt.get_cmap('tab10')(i))
    for j in range(5):
        plt.plot(xs, vals[xs, j])
    plt.xticks(xs)
    plt.title(label)
    plt.show()

# ======== cohort explanation ========
k_best = 4
idx = np.load(f'./output/shap_all/idx_test_k={k_best}_t=0.npy')
labels = np.load(f'./output/shap_all/labels_k={k_best}_t=0.npy')
descriptions = np.load(f'./output/shap_all/descriptions_k={k_best}_t=0.npy')

importance = np.load('data/inflammation/importance_shap.npy')
importance = importance[idx]
correct = correct[idx]

fig, axs = plt.subplots(3, k_best, dpi=300, figsize=(30, 15), sharex='all')
for i in range(k_best):
    members = importance[labels == i]
    print(f'cohort {i}')
    print('number of samples:', len(members))
    print(f'descriptions: {discrete_features[descriptions[i]]}')

    df = pd.DataFrame(importance[labels == i][:, food_features_idx],
                      columns=[x_test.columns[j] for j in food_features_idx])
    sns.barplot(df, orient='h', ax=axs[0, i])
    axs[0, i].set_title(f'descriptions: {discrete_features[descriptions[i]]}\n'
                        f'number of samples: {len(members)}\n'
                        f'acc: {np.mean(correct[labels == i]):.4f}\n'
                        f'overall')

    df = pd.DataFrame(importance[labels == i][correct[labels == i]][:, food_features_idx],
                      columns=[x_test.columns[j] for j in food_features_idx])
    sns.barplot(df, orient='h', ax=axs[1, i])
    axs[1, i].set_title(f'correct samples'
                        f'number of samples: {np.count_nonzero(correct[labels == i])}\n')

    df = pd.DataFrame(importance[labels == i][np.bitwise_not(correct)[labels == i]][:, food_features_idx],
                      columns=[x_test.columns[j] for j in food_features_idx])
    sns.barplot(df, orient='h', ax=axs[2, i])
    axs[2, i].set_title(f'incorrect samples'
                        f'number of samples: {np.count_nonzero(np.bitwise_not(correct)[labels == i])}\n')
plt.tight_layout()
plt.show()

# plot differences
avg_food_importance = np.mean(importance[:, food_features_idx], axis=0)
fig, axs = plt.subplots(1, k_best, dpi=300, figsize=(20, 6), sharex='all')
for i in range(k_best):
    members = importance[labels == i]
    print(f'cohort {i}')
    print('number of samples:', len(members))
    print(f'descriptions: {discrete_features[descriptions[i]]}')
    diff = np.mean(importance[labels == i][:, food_features_idx], axis=0) - avg_food_importance
    df = pd.DataFrame(importance[labels == i][:, food_features_idx] - avg_food_importance,
                      columns=[x_test.columns[i] for i in food_features_idx])
    sns.barplot(df, orient='h', ax=axs[i], hatch='//', fill=False)
    axs[i].set_title(f'descriptions: {discrete_features[descriptions[i]]}\n'
                     f'number of samples: {len(members)}')
plt.tight_layout()
plt.show()
