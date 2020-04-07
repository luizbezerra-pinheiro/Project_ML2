# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

from src.GetData import GetData
from src.FeatureEngineering import FeatEng
from src.OurModel import OurModel
from src.Evaluation import Evaluation

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


if __name__ == "__main__":
    # Get data
    df_train, df_test = GetData().get()

    # Feature Engineering of df_train
    myFE = FeatEng(selected_feat=True)
    myFE.fit(df_train)
    X, y = myFE.transform(df_train)

    ### ----- Cross-validation for the df_train
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
        eval_over = Evaluation(myModel_over, X_over, y_over, X_test, y_test, verbatim=False)
        perf_over.append(eval_over.eval)
        eval_under = Evaluation(myModel_under, X_under, y_under, X_test, y_test, verbatim=False)
        perf_under.append(eval_under.eval)

    perf_over = np.array(perf_over)
    perf_under = np.array(perf_under)

    perf_over = perf_over.mean(axis=0)
    perf_under = perf_under.mean(axis=0)

    print("### Cross Validation:")
    print("\t## Oversampling")
    name_models = ["RandomForestClassifier", "LogisticRegression"]
    for i, name in zip(range(2), name_models):
        print("\t\t#", name)
        print("\t\t\tTrain performance:", perf_over[i,0])
        print("\t\t\tTrain f1-score:", perf_over[i,1])
        print("\t\t\tTest performance:", perf_over[i, 2])
        print("\t\t\tTest f1-score:", perf_over[i, 3])
    print("\t## Undersampling")
    for i, name in zip(range(2), name_models):
        print("\t\t#", name)
        print("\t\t\tTrain performance:", perf_under[i,0])
        print("\t\t\tTrain f1-score:", perf_under[i,1])
        print("\t\t\tTest performance:", perf_under[i, 2])
        print("\t\t\tTest f1-score:", perf_under[i, 3], "\n")


    # Modeling
    X_train, y_train = X, y

    # Oversampling and Undersampling
    X_over, y_over = oversample.fit_resample(X_train, y_train)
    X_under, y_under = undersample.fit_resample(X_train, y_train)

    # Modeling
    myModel_over = OurModel()
    myModel_over.fit(X_over, y_over)

    myModel_under = OurModel()
    myModel_under.fit(X_under, y_under)

    # Preprocessing the test data
    X_test, y_test = myFE.transform(df_test)

    # Evaluation
    eval_over = Evaluation(myModel_over, X_over, y_over, X_test, y_test, verbatim=True)
    eval_under = Evaluation(myModel_under, X_under, y_under, X_test, y_test, verbatim=True)

    #
    #
    # # Split train and test
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
    #
    # print('Training Features Shape:', X_train.shape)
    # print('Training Labels Shape:', y_train.shape)
    # print('Testing Features Shape:', X_test.shape)
    # print('Testing Labels Shape:', y_test.shape)
    #
    # # Fitting the model
    # clf = RandomForestClassifier(n_estimators=1000, random_state=0)
    # clf2 = LogisticRegression(random_state=0)
    #
    # scores = cross_val_score(clf, X_train, y_train, cv=5)
    # print(scores.mean())
    # scores2 = cross_val_score(clf2, X_train, y_train, cv=5)
    # print(scores2.mean())
    #
    # clf.fit(X_train, y_train)
    # clf2.fit(X_train, y_train)
    #
    # # Predict
    # y_pred = clf.predict(X_test)
    # y_train_pred = clf.predict(X_train)
    #
    # y_pred2 = clf2.predict(X_test)
    # y_train_pred2 = clf2.predict(X_train)
    #
    #
    # # Performance
    # #print(confusion_matrix(y_test, y_pred))
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    # print("\nRandomForest:\nTest\n\tPerformance test:", (tp-fp)/(tp+fn))
    # print("\tf1-score: ", f1_score(y_test, y_pred))
    #
    # #print(confusion_matrix(y_train, y_train_pred))
    # tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    # print("\nTrain\n\tPerformance train:", (tp - fp) / (tp + fn))
    # print("\tf1-score: ", f1_score(y_train, y_train_pred))
    #
    # #print(confusion_matrix(y_test, y_pred2))
    # tn, fp, fn, tp = confusion_matrix(y_test, y_pred2).ravel()
    # print("\nLogisticRegression:\nTest\n\tPerformance:", (tp - fp) / (tp + fn))
    # print("\tf1-score: ", f1_score(y_test, y_pred2))
    #
    # #print(confusion_matrix(y_train, y_train_pred2))
    # tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred2).ravel()
    # print("\nTrain\n\tPerformance train:", (tp - fp) / (tp + fn))
    # print("\tf1-score: ", f1_score(y_train, y_train_pred2))


