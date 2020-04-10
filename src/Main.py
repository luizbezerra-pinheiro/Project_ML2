# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

from src.GetData import GetData
from src.FeatureEngineering import FeatEng
from src.OurModel import OurModel
from src.Evaluation import Evaluation
from src.DataAnalysis import DataAnalysis

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from tqdm import tqdm

if __name__ == "__main__":
    # Get data
    df_train, df_test = GetData().get()

    # Feature Engineering of df_train
    myFE = FeatEng()
    myFE.fit(df_train)
    # X, y, feature_names = myFE.transform(df_train)
    df_train = myFE.transform(df_train)

    DataAnalysis(df_train, "1_train_featEng")

    ### ----- Cross-validation for the df_train
    X = np.array(df_train.drop(['Y'], axis=1))
    y = np.array(df_train[['Y']]).reshape(-1, )
    feature_names = df_train.drop(['Y'], axis=1).columns

    skf = StratifiedKFold(n_splits=5)
    skf.get_n_splits(X, y)

    perf_over = []
    perf_under = []
    # For each split
    # i = 1
    for train_index, test_index in skf.split(X, y):
        # print("\n###", i, "\n")
        # i += 1

        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]

        # Treating the imbalanced data
        # Oversampling and Undersampling
        oversample = RandomOverSampler(sampling_strategy=1)
        undersample = RandomUnderSampler(sampling_strategy=1)

        X_over, y_over = oversample.fit_resample(X_train, y_train)
        X_under, y_under = undersample.fit_resample(X_train, y_train)

        # Modeling
        myModel_over = OurModel()
        myModel_over.fit(X_over, y_over)

        myModel_under = OurModel()
        myModel_under.fit(X_under, y_under)

        #Evaluation of the models
        eval_over = Evaluation(myModel_over, X_over, y_over, X_test, y_test, verbose=False)
        perf_over.append(eval_over.eval)
        eval_under = Evaluation(myModel_under, X_under, y_under, X_test, y_test, verbose=False)
        perf_under.append(eval_under.eval)

    perf_over = np.array(perf_over)
    perf_under = np.array(perf_under)

    perf_over = perf_over.mean(axis=0)
    perf_under = perf_under.mean(axis=0)

    print("### Cross Validation:")
    print("\t## Oversampling")
    name_models = [type(m).__name__ for m in OurModel().models]
    for i, name in enumerate(name_models):
        print("\t\t#", name)
        print("\t\t\tTrain performance:", perf_over[i,0])
        print("\t\t\tTrain f1-score:", perf_over[i,1])
        print("\t\t\tTest performance:", perf_over[i, 2])
        print("\t\t\tTest f1-score:", perf_over[i, 3])
    print("\t## Undersampling")
    for i, name in enumerate(name_models):
        print("\t\t#", name)
        print("\t\t\tTrain performance:", perf_under[i,0])
        print("\t\t\tTrain f1-score:", perf_under[i,1])
        print("\t\t\tTest performance:", perf_under[i, 2])
        print("\t\t\tTest f1-score:", perf_under[i, 3], "\n")


    # Modeling
    X_train, y_train = X, y

    # Oversampling and Undersampling
    oversample = RandomOverSampler(sampling_strategy=1)
    undersample = RandomUnderSampler(sampling_strategy=1)

    X_over, y_over = oversample.fit_resample(X_train, y_train)
    X_under, y_under = undersample.fit_resample(X_train, y_train)

    # Data Analysis
    aux = df_train[0:0]
    type_dic = dict(aux.dtypes)
    aux[feature_names] = X_over
    aux["Y"] = y_over
    aux = aux.astype(type_dic)
    DataAnalysis(aux, "2_train_oversampling")

    # Modeling
    myModel_over = OurModel()
    myModel_over.fit(X_over, y_over)

    print("TESTEEE 2\n")
    print(myModel_over.models[0].feature_importances_)
    plt.barh(range(X_over.shape[1]), myModel_over.models[0].feature_importances_, align='center')
    plt.yticks(range(X_over.shape[1]), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()

    myModel_under = OurModel()
    myModel_under.fit(X_under, y_under)

    # Preprocessing the test data
    df_test = myFE.transform(df_test)

    X_test = np.array(df_test.drop(['Y'], axis=1))
    y_test = np.array(df_test[['Y']]).reshape(-1, )

    # Evaluation
    eval_over = Evaluation(myModel_over, X_over, y_over, X_test, y_test, verbose=True)
    eval_under = Evaluation(myModel_under, X_under, y_under, X_test, y_test, verbose=True)


