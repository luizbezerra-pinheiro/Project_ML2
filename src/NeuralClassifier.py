from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator


class NeuralClassifier(BaseEstimator):
    def __init__(self, epochs=10, batch_size=10, verbose=2, optimizer='adam'):
        super(NeuralClassifier, self).__init__()
        self.input_dim = None
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.optimizer = optimizer

    def fit(self, X, y, class_weights=None):
        # n_samples, n_features = X.shape
        onehot_y = OneHotEncoder().fit_transform(y.reshape(-1, 1))
        self.input_dim = X.shape[1]
        self.model = Sequential()
        self.model.add(Dense(16, input_dim=self.input_dim, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(4, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))
        # Compile model
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        # Fit the model
        self.model.fit(X, onehot_y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                       class_weight=class_weights)

    def predict(self, X):
        return np.array([1 if y[1] > 0.5 else 0 for y in self.model.predict(X)])


if __name__ == '__main__':
    exit(0)
