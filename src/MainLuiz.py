# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

from src.FeatureEngineeringLuiz import FeatureEngineering
import numpy as np
import pandas as pd
import os
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score

class RandomForest(FeatureEngineering):
    def __init__(self):
        super().__init__()


def hypothetical_revenue(truth, predictions):
    TP=0
    FP=0
    FN=0
    for y, p in zip(truth, predictions):
        if y==1 and p==1:
            TP += 1
        elif y==0 and p==1:
            FP += 1
        elif y==1 and p==0:
            FN +=1
    return (TP-FP)/(TP+FN) if TP+FN > 0 else 0

if __name__ == "__main__":
    ## Pre-processing

    # Define features to be selected
    selected_features = ["Y", "Customer_Type", "P_Client", "Educational_Level", "Marital_Status", "Number_Of_Dependant",
                         "Years_At_Residence", "Net_Annual_Income", "Years_At_Business",
                         "Source", "Type_Of_Residence", "Nb_Of_Products"]
    # Define categorical_features to be converted
    categorical_features = ['Y', 'Customer_Type', 'Educational_Level', 'Marital_Status',
                            'P_Client', 'Source', 'Type_Of_Residence']

    # Numerical Features
    numerical_features = {'float64': ['Net_Annual_Income'],
                          'int64': ['Number_Of_Dependant', 'Years_At_Residence',
                                    'Years_At_Business', 'Nb_Of_Products']}

    data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "CreditTraining.csv")
    fe = FeatureEngineering(pd.read_csv(data_path, decimal=','))
    fe.transform(selected_features=selected_features, categorical_features=categorical_features,
                 numerical_features=numerical_features, one_hot=True)

    # Random-Forest model
    # print(fe.df.describe())
    labels = np.array(fe.df[['Y_0', 'Y_1']])
    fe.df = fe.df.drop(['Y_0', 'Y_1'], axis=1)
    features_names = list(fe.df)

    # Convert to numpy array
    features = np.array(fe.df)

    #### End pre-processing

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    ### Testing Models

    # Random Forest
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Test predictions
    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    predictions = np.array([0 if x[0] > 0.5 else 1 for x in predictions])
    test_labels = np.array([0 if x[0] > 0.5 else 1 for x in test_labels])
    # Calculate the absolute errors

    print('f1_score: ', f1_score(test_labels, predictions))
    print('hypothetical revenue: ', hypothetical_revenue(test_labels, predictions))
    #errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    #print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features_names, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
