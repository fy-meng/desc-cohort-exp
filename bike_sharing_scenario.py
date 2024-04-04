import os

import numpy as np
import pandas
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from ucimlrepo import fetch_ucirepo
import xgboost as xgb


class BikeScenario:
    # parameters: {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 150}
    # R2 score: 0.9453162320502967
    def __init__(self):
        # bike_sharing_dataset = fetch_ucirepo(id=275)
        # self.X_train = bike_sharing_dataset.data.features.copy()
        # self.y_train = bike_sharing_dataset.data.targets
        dataset = pandas.read_csv('data/bike/bike.csv')
        self.X_train = dataset[
            ['dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp',
             'hum', 'windspeed']]
        self.y_train = dataset['cnt']

        self.feature_names = ['date', 'season', 'year', 'month', 'hour', 'is_holiday', 'weekday',
                              'is_workday', 'weather', 'temp', 'feeling_temp',
                              'humidity', 'wind_speed']
        self.features = self.X_train.columns

        # data cleaning
        self.X_train['dteday'] = pd.to_datetime(self.X_train['dteday'])
        self.X_train['dteday'] = self.X_train['dteday'].apply(lambda x: x.value) // (10 ** 9)
        self.categorical_features = ['season', 'holiday', 'weekday', 'weathersit', 'workingday', 'mnth', 'yr', 'hr']
        for col in self.categorical_features:
            self.X_train[col] = self.X_train[col].astype('category')

        n_samples = 1000
        np.random.seed(0)
        self.X_train = self.X_train.apply(np.random.permutation, axis=0)
        self.X = self.X_train[:n_samples].values
        self.y = self.y_train[:n_samples].values

        model_path = 'data/bike/bike_sharing.json'
        if os.path.exists(model_path):
            self.model = xgb.XGBRegressor(enable_categorical=True, tree_method='hist')
            self.model.load_model(model_path)
        else:
            self.model = xgb.XGBRegressor(enable_categorical=True, tree_method='hist')
            self.train()
            self.model.save_model(model_path)

    def train(self):
        # Define the hyperparameters for tuning
        params = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.01, 0.001]
        }

        # perform hyperparameter tuning using gridsearch
        grid_search = GridSearchCV(self.model, params, cv=5, scoring='r2')
        grid_search.fit(self.X_train, self.y_train)

        print('Best hyperparameters:', grid_search.best_params_)

        self.model = grid_search.best_estimator_
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_train)

        r2 = r2_score(self.y_train, y_pred)
        print(f'r2 = {r2}')


if __name__ == '__main__':
    s = BikeScenario()
