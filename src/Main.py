# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

from src.GetData import GetData
from src.FeatureEngineering import FeatEng
from src.OurModel import OurModel
from src.Evaluation import Evaluation
from src.DataAnalysis import DataAnalysis
from src.TuningHyperparameters import TuningHyperparameters

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import json


from sklearn.model_selection import StratifiedKFold
import prince

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from tqdm import tqdm

if __name__ == "__main__":
    # Get data
    df_train, df_test = GetData().get()

    ### ----- Cross-validation for the df_train
    X = np.array(df_train.drop(['Y'], axis=1))
    y = np.array(df_train[['Y']]).reshape(-1, )
    feature_names = df_train.drop(['Y'], axis=1).columns

    DataAnalysis(df_train, "1_train_featEng")

    ### ----- Cross-validation for the df_train
    skf = StratifiedKFold(n_splits=5)

    perf_over = []
    perf_under = []
    perf_direct = []
    # For each split
    for train_index, test_index in tqdm(skf.split(X, y)):
        df_train_cv = df_train.iloc[train_index].reset_index(drop=True)
        df_test_cv = df_train.iloc[test_index].reset_index(drop=True)

        # FeatEng
        myFE = FeatEng()
        df_train_cv = myFE.fit_transform(df_train_cv)
        df_test_cv = myFE.transform(df_test_cv)

        # print(df_train_cv.describe(include='all'))

        X_train, y_train = np.array(df_train_cv.drop(['Y'], axis=1)), np.array(df_train_cv["Y"])
        X_test, y_test = np.array(df_test_cv.drop(['Y'], axis=1)), np.array(df_test_cv["Y"])

        # Treating the imbalanced data
        # Oversampling and Undersampling
        oversample = RandomOverSampler(sampling_strategy=1)
        undersample = RandomUnderSampler(sampling_strategy=1)

        X_over, y_over = oversample.fit_resample(X_train, y_train)
        X_under, y_under = undersample.fit_resample(X_train, y_train)

        # Modeling
        myModel_over = OurModel(sampling_mode='over')
        myModel_over.fit(X_over, y_over)

        myModel_under = OurModel(sampling_mode='under')
        myModel_under.fit(X_under, y_under)

        myModel_direct = OurModel()
        myModel_direct.fit(X_train, y_train)

        # Evaluation of the models
        eval_over = Evaluation(myModel_over, X_over, y_over, X_test, y_test, verbose=False)
        perf_over.append(eval_over.eval)
        eval_under = Evaluation(myModel_under, X_under, y_under, X_test, y_test, verbose=False)
        perf_under.append(eval_under.eval)
        eval_direct = Evaluation(myModel_direct, X_train, y_train, X_test, y_test, verbose=False)
        perf_direct.append(eval_direct.eval)

    perf_over = np.array(perf_over)
    perf_under = np.array(perf_under)
    perf_direct = np.array(perf_direct)

    perf_over = perf_over.mean(axis=0)
    perf_under = perf_under.mean(axis=0)
    perf_direct = perf_direct.mean(axis=0)

    print("### Cross Validation:")
    print("\t## Oversampling")
    name_models = [type(m).__name__ for m in OurModel().models]
    for i, name in enumerate(name_models):
        print("\t\t#", name)
        print("\t\t\tTrain performance:", perf_over[i, 0])
        print("\t\t\tTrain f1-score:", perf_over[i, 1])
        print("\t\t\tTest performance:", perf_over[i, 2])
        print("\t\t\tTest f1-score:", perf_over[i, 3])
    print("\t## Undersampling")
    for i, name in enumerate(name_models):
        print("\t\t#", name)
        print("\t\t\tTrain performance:", perf_under[i, 0])
        print("\t\t\tTrain f1-score:", perf_under[i, 1])
        print("\t\t\tTest performance:", perf_under[i, 2])
        print("\t\t\tTest f1-score:", perf_under[i, 3], "\n")
    print("\t## Direct")
    for i, name in enumerate(name_models):
        print("\t\t#", name)
        print("\t\t\tTrain performance:", perf_direct[i, 0])
        print("\t\t\tTrain f1-score:", perf_direct[i, 1])
        print("\t\t\tTest performance:", perf_direct[i, 2])
        print("\t\t\tTest f1-score:", perf_direct[i, 3], "\n")

    # Modeling

    # Feature Engineering of df_train
    myFE = FeatEng()
    df_train = myFE.fit_transform(df_train)

    # Analysis of the Data Train
    DataAnalysis(df_train, "1_train_featEng")

    X_train = np.array(df_train.drop(['Y'], axis=1))
    y_train = np.array(df_train[['Y']]).reshape(-1, )

    # Oversampling and Undersampling
    oversample = RandomOverSampler(sampling_strategy=1)
    undersample = RandomUnderSampler(sampling_strategy=1)

    X_over, y_over = oversample.fit_resample(X_train, y_train)
    X_under, y_under = undersample.fit_resample(X_train, y_train)

    # Data Analysis
    aux = df_train[0:0]
    feature_names = [c for c in df_train.columns if c != "Y"]
    type_dic = dict(aux.dtypes)
    aux[feature_names] = X_over
    aux["Y"] = y_over
    aux = aux.astype(type_dic)
    DataAnalysis(aux, "2_train_oversampling")

    # Modeling
    myModel_over = OurModel(sampling_mode='over')
    myModel_over.fit(X_over, y_over)

    print("TESTEEE 2\n")
    print(myModel_over.models[0].feature_importances_)
    plt.barh(range(X_over.shape[1]), myModel_over.models[0].feature_importances_, align='center')
    plt.yticks(range(X_over.shape[1]), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.show()

    myModel_under = OurModel(sampling_mode='under')
    myModel_under.fit(X_under, y_under)

    myModel_direct = OurModel()
    myModel_direct.fit(X_train, y_train)

    # Preprocessing the test data
    df_test = myFE.transform(df_test)

    X_test = np.array(df_test.drop(['Y'], axis=1))
    y_test = np.array(df_test[['Y']]).reshape(-1, )

    # Evaluation
    eval_over = Evaluation(myModel_over, X_over, y_over, X_test, y_test, verbose=True)
    eval_under = Evaluation(myModel_under, X_under, y_under, X_test, y_test, verbose=True)
    eval_direct = Evaluation(myModel_direct, X_train, y_train, X_test, y_test, verbose=True)
