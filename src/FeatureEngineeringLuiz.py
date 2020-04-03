import os

import pandas as pd
from copy import copy


class FeatureEngineering:
    def __init__(self, dataframe):
        self.df = dataframe

    def filter_variables(self, selected_features):
        self.df = self.df[selected_features].fillna(0)

    def convert_categorical(self, categorical_features):
        # Use LabelEncoder from sklearn to convert the categorical variables into numerical
        for column in categorical_features:
            # converting type of columns to 'category'
            self.df[column] = self.df[column].astype('category')
            # Assigning numerical values and storing in another column
            self.df[column] = self.df[column].cat.codes
        self.df[categorical_features] = self.df[categorical_features].astype('category')

    def cast_numerical(self, numerical_features):
        # Float values
        self.df[numerical_features['float64']] = self.df[numerical_features['float64']].astype('float64')

        # Integer values
        self.df[numerical_features['int64']] = self.df[numerical_features['int64']].astype('int64')

    def one_hot_encoding(self):
        self.df = pd.get_dummies(self.df)

    def transform(self, one_hot=False, selected_features=None, categorical_features=None, numerical_features=None):
        if selected_features:
            self.filter_variables(selected_features)
        if categorical_features:
            self.convert_categorical(categorical_features)
        if numerical_features:
            self.cast_numerical(numerical_features)
        if one_hot:
            self.one_hot_encoding()


## Tests
if __name__ == "__main__":
    print(os.path.dirname(os.getcwd()), os.path.dirname('.'))
    data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "CreditTraining.csv")
    fe = FeatureEngineering(pd.read_csv(data_path, decimal=','))

    # Define features to be selected
    unwanted_features = ["Prod_Sub_Category", "Prod_Category"]
    selected_features = ["Y", "Customer_Type", "P_Client", "Educational_Level", "Marital_Status", "Number_Of_Dependant",
                         "Years_At_Residence", "Net_Annual_Income", "Years_At_Business",
                         "Source", "Type_Of_Residence", "Nb_Of_Products"]
    # Define categorical_features to be converted
    categorical_features = ['Y', 'Customer_Type', 'Educational_Level', 'Marital_Status',
                            'P_Client', 'Prod_Category', 'Prod_Sub_Category',
                            'Source', 'Type_Of_Residence']

    # Numerical Features
    numerical_features = {'float64': ['Net_Annual_Income'],
                          'int64': ['Number_Of_Dependant', 'Years_At_Residence',
                                    'Years_At_Business', 'Nb_Of_Products']}

    fe.transform(selected_features=selected_features, categorical_features=categorical_features,
                 numerical_features=numerical_features, one_hot=True)
    print(fe.df.dtypes)
    fe.df.to_csv('../data/Credit.csv', index=False)
    print(fe.df.head())
