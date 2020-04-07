from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class OurModel:
    def __init__(self):
        self.models = [
            RandomForestClassifier(n_estimators=1000, random_state=0),
            LogisticRegression(random_state=0)
        ]

    def fit(self, X_train, y_train):
        for m in self.models:
            m.fit(X_train, y_train)

    def predict(self, X_test):
        y_pred = []
        for m in self.models:
            y_pred.append(m.predict(X_test))
        return y_pred


## Tests
if __name__ == "__main__":
    exit()
