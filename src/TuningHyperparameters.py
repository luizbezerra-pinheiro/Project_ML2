from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, make_scorer
from src.GetData import GetData
from src.OurModel import OurModel
from src.FeatureEngineering import FeatEng
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import json


def my_custom_loss_func(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (tp - fp) / (tp + fn)


perf_scorer = make_scorer(my_custom_loss_func, greater_is_better=True)


class TuningHyperparameters:

    def __init__(self, models, params, verbose=True):
        self.models = models
        self.params = params
        self.models_rand = []
        self.verbose=verbose
        for model, grid in zip(self.models, self.params):
            self.models_rand.append(
                RandomizedSearchCV(estimator=model, param_distributions=grid, scoring=perf_scorer, n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1))

    def fit(self, X, y):
        for model_rand in self.models_rand:
            model_rand.fit(X, y)
            if self.verbose:
                print(f'Best score: {model_rand.best_score_}')

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
    # Modeling
    # Get data
    df_train, df_test = GetData().get()

    # Feature Engineering of df_train
    myFE = FeatEng()
    df_train = myFE.fit_transform(df_train)

    X_train = np.array(df_train.drop(['Y'], axis=1))
    y_train = np.array(df_train[['Y']]).reshape(-1, )

    # Oversampling and Undersampling
    oversample = RandomOverSampler(sampling_strategy=1)
    undersample = RandomUnderSampler(sampling_strategy=1)

    X_over, y_over = oversample.fit_resample(X_train, y_train)
    X_under, y_under = undersample.fit_resample(X_train, y_train)

    # Hyperparameter tuning
    # Direct
    myModel = OurModel()
    models_tune = TuningHyperparameters(myModel.models, myModel.params_to_tune)
    models_tune.fit(X_train, y_train)
    print(models_tune.get_best_params())
    choosen_params = models_tune.get_best_params()
    with open('hyperparameters_direct.json', 'w') as f:
        json.dump(choosen_params, f)

    # Oversampling
    myModel = OurModel()
    models_tune = TuningHyperparameters(myModel.models, myModel.params_to_tune)
    models_tune.fit(X_over, y_over)
    print(models_tune.get_best_params())
    choosen_params = models_tune.get_best_params()
    with open('hyperparameters_over.json', 'w') as f:
        json.dump(choosen_params, f)

    # Undersampling
    myModel = OurModel()
    models_tune = TuningHyperparameters(myModel.models, myModel.params_to_tune)
    models_tune.fit(X_under, y_under)
    print(models_tune.get_best_params())
    choosen_params = models_tune.get_best_params()
    with open('hyperparameters_under.json', 'w') as f:
        json.dump(choosen_params, f)

