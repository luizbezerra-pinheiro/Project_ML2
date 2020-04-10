from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def create_model(input_dim):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class NeuralClassifier:
    def __init__(self, epochs=10, batch_size=10, verbose=1):
        super(NeuralClassifier, self).__init__()
        self.input_dim = None
        self.model = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X, y):
        # n_samples, n_features = X.shape
        self.input_dim = X.shape[1]
        self.model = Sequential()
        self.model.add(Dense(16, input_dim=self.input_dim, activation='relu'))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dense(4, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        # Compile model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, X):
        return np.array([1 if y > 0.5 else 0 for y in self.model.predict(X)])
