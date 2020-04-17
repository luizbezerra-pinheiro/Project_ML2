from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.NeuralClassifier import NeuralClassifier
import numpy as np
import json
import os

params_dir = os.path.join(os.path.dirname(os.getcwd()), "params")


class OurModel:
    def __init__(self, sampling_mode=None):
        self.params = self.parameters(sampling_mode)

        self.models = [
            RandomForestClassifier(random_state=0, **self.params[0]),  # n_estimators=500, random_state=0,
            LogisticRegression(random_state=0, **self.params[1]),
            #NeuralClassifier(**params[2])  # epochs=100, batch_size=16,
        ]

        self.params_to_tune = [
            {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
             # Number of trees in random forest
             'max_features': ['auto', 'sqrt'],  # Number of features to consider at every split
             'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],  # Maximum number of levels in tree
             'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
             'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
             'bootstrap': [True, False]},  # Create the random grid
            {'max_iter': [1000, 2000, 2500]},
            {'epochs': [int(x) for x in np.linspace(10, 110, num=11)],
             'batch_size': [8, 16, 32, 64],
             'optimizer': ['adam', 'sgd']}
        ]

    def fit(self, X_train, y_train):
        for m in self.models:
            m.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = []
        for m in self.models:
            y_pred.append(m.predict(X_test))
        return y_pred

    def parameters(self, sampling_mode):
        params = None
        if sampling_mode == 'over':
            try:
                with open(os.path.join(params_dir, 'hyperparameters_over.json'), 'r') as f:
                    params = json.load(f)
                    # print('loading oversampling hyperparameters:\n', params)
            except FileNotFoundError:
                pass
        elif sampling_mode == 'under':
            try:
                with open(os.path.join(params_dir, 'hyperparameters_under.json'), 'r') as f:
                    params = json.load(f)
                    # print('loading undersampling hyperparameters:\n', params)
            except FileNotFoundError:
                pass
        else:
            try:
                with open(os.path.join(params_dir, 'hyperparameters_direct.json'), 'r') as f:
                    params = json.load(f)
                    # print('loading directsampling hyperparameters:\n', params)
            except FileNotFoundError:
                pass

        if params is None:
            params = [{"n_estimators": 500}, {}, {"epochs": 100, "batch_size": 16}]

        return params



## Tests
if __name__ == "__main__":
    exit()
