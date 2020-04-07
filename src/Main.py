# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

from src.GetData import GetData
from src.FeatureEngineering import FeatEng

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    # Get data
    df_train, df_test = GetData().get()

    # Feature Engineering
    myFE = FeatEng(df_train)
    X, y = myFE.transform()

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

    print('Training Features Shape:', X_train.shape)
    print('Training Labels Shape:', y_train.shape)
    print('Testing Features Shape:', X_test.shape)
    print('Testing Labels Shape:', y_test.shape)

    # Fitting the model
    clf = RandomForestClassifier(n_estimators=1000, random_state=0)
    clf2 = LogisticRegression(random_state=0)

    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(scores.mean())
    scores2 = cross_val_score(clf2, X_train, y_train, cv=5)
    print(scores2.mean())

    clf.fit(X_train, y_train)
    clf2.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)

    y_pred2 = clf2.predict(X_test)
    y_train_pred2 = clf2.predict(X_train)


    # Performance
    #print(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("\nRandomForest:\nTest\n\tPerformance test:", (tp-fp)/(tp+fn))
    print("\tf1-score: ", f1_score(y_test, y_pred))

    #print(confusion_matrix(y_train, y_train_pred))
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    print("\nTrain\n\tPerformance train:", (tp - fp) / (tp + fn))
    print("\tf1-score: ", f1_score(y_train, y_train_pred))

    #print(confusion_matrix(y_test, y_pred2))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred2).ravel()
    print("\nLogisticRegression:\nTest\n\tPerformance:", (tp - fp) / (tp + fn))
    print("\tf1-score: ", f1_score(y_test, y_pred2))

    #print(confusion_matrix(y_train, y_train_pred2))
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred2).ravel()
    print("\nTrain\n\tPerformance train:", (tp - fp) / (tp + fn))
    print("\tf1-score: ", f1_score(y_train, y_train_pred2))


