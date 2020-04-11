from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint# Look at parameters used by our current forest
import numpy as np

class TuningHyperparameters:

    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.models_rand = []
        for model, grid in zip(self.models, self.params):
            self.models_rand.append(RandomizedSearchCV(estimator=model, param_distributions=grid, n_iter=100, cv=5, verbose=2,
                                                       random_state=42, n_jobs=-1))

    def fit(self, X, y):
        for model_rand in self.models_rand:
            model_rand.fit(X, y)

    def transform(self):
        best_models = []
        for model_rand in self.models_rand:
            best_models.append(model_rand.best_estimator_)
        return best_models

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform()

    def get_best_params(self):
        best_params = []
        for model_rand in self.models_rand:
           best_params.append(model_rand.best_params_)
        return best_params


if __name__ == '__main__':
    rf = RandomForestRegressor()

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]  # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_rand = TuningHyperparameters([rf], [random_grid])

   # best_rf_rand = rf_rand.fit_transform()

    exit()