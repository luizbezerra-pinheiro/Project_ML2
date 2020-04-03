from src.GetData import GetData
from src.FeatureEngineering import FeatEng

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":
    # Get data
    df_train, df_test = GetData().get()

    # Feature Engineering
    myFE = FeatEng()
    X, y = myFE.transform(df_train)

    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)

    # print(len(X_train), len(y_train))
    # print(X_train)

    # Fitting the model
    clf = RandomForestClassifier(n_estimators=1000, max_depth=2, random_state=0)
    clf2 = LogisticRegression(random_state=0)

    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(scores.mean())
    scores2 = cross_val_score(clf2, X_train, y_train, cv=5)
    print(scores.mean())

    clf.fit(X_train, y_train)
    clf2.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)

    y_pred2 = clf.predict(X_test)
    y_train_pred2 = clf.predict(X_train)


    # Performance
    print(confusion_matrix(y_test, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("\nPerformance:", (tp-fp)/(tp+fn))

    print(confusion_matrix(y_train, y_train_pred))
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    print("\nPerformance:", (tp - fp) / (tp + fn))

    print(confusion_matrix(y_test, y_pred2))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred2).ravel()
    print("\nPerformance:", (tp - fp) / (tp + fn))

    print(confusion_matrix(y_train, y_train_pred2))
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred2).ravel()
    print("\nPerformance:", (tp - fp) / (tp + fn))


