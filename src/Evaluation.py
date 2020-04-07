from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


class Evaluation:
    def __init__(self, ourModel, X_train, y_train, X_test, y_test, verbatim=True):
        self.ourModel = ourModel
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.verbatim = verbatim

        self.eval = []

        self.print_evaluation()


    """
    Print all the performance and the f1-score
    """
    def print_evaluation(self):
        if self.verbatim:
            print("\n#### EVALUATION\n")
        for m in self.ourModel.models:
            perf_f1 = []
            if self.verbatim:
                print("\t###", type(m).__name__)

            y_train_pred = m.predict(self.X_train)
            y_test_pred = m.predict(self.X_test)

            # Confusion Matrix - Train:

            tn, fp, fn, tp = confusion_matrix(self.y_train, y_train_pred).ravel()
            perf_f1.append((tp - fp) / (tp + fn))
            perf_f1.append(f1_score(self.y_train, y_train_pred))
            if self.verbatim:
                print("\t\t# Train")
                print("\t\t\tPerformance :", perf_f1[-2])
                print("\t\t\tf1-score:", perf_f1[-1])


            # Confusion Matrix - Test:

            tn, fp, fn, tp = confusion_matrix(self.y_test, y_test_pred).ravel()
            perf_f1.append((tp - fp) / (tp + fn))
            perf_f1.append(f1_score(self.y_test, y_test_pred))
            if self.verbatim:
                print("\t\t# Test")
                print("\t\t\tPerformance :", perf_f1[-2])
                print("\t\t\tf1-score:", perf_f1[-1], "\n")

            self.eval.append(perf_f1)


## Tests
if __name__ == "__main__":
    exit()
