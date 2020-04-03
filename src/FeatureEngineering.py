import os

import pandas as pd
import numpy as np


class FeatEng:
    def __init__(self, dataframe):
        self.df = dataframe
        # self.selected_features = ["Y", "Customer_Type", "P_Client", "Educational_Level", "Marital_Status",
        #                          "Number_Of_Dependant", "Years_At_Residence", "Net_Annual_Income",
        #                          "Years_At_Business", "Source", "Type_Of_Residence", "Nb_Of_Products"]
        self.selected_features = ["Y", "Net_Annual_Income", "Years_At_Business",
                                  "Years_At_Residence", "Number_Of_Dependant"]
        self.categorical_features = ['Y', 'Customer_Type', 'Educational_Level', 'Marital_Status',
                                     'P_Client', 'Prod_Category', 'Prod_Sub_Category',
                                     'Source', 'Type_Of_Residence']
        self.numerical_features = {'float64': ['Net_Annual_Income'],
                                   'int64': ['Number_Of_Dependant', 'Years_At_Residence',
                                             'Years_At_Business', 'Nb_Of_Products']}
        self.datetime_variables = ['BirthDate', 'Customer_Open_Date', 'Prod_Decision_Date', 'Prod_Closed_Date']
        self.one_hot = True

    def filter_variables(self):
        self.df = self.df[self.selected_features].fillna(0)

    def convert_categorical(self):
        # Use LabelEncoder from sklearn to convert the categorical variables into numerical
        for column in self.categorical_features:
            if column in self.selected_features:
                # converting type of columns to 'category'
                self.df[column] = self.df[column].astype('category')
                # Assigning numerical values and storing in another column
                self.df[column] = self.df[column].cat.codes.astype('category')

    def cast_numerical(self):

        selected_numerical_float = list(
            set(self.numerical_features['float64']).intersection(set(self.selected_features)))
        selected_numerical_int = list(set(self.numerical_features['int64']).intersection(set(self.selected_features)))
        # Float values
        self.df[selected_numerical_float] = self.df[selected_numerical_float].astype('float64')

        # Integer values
        self.df[selected_numerical_int] = self.df[selected_numerical_int].astype('int64')

    def one_hot_encoding(self):
        self.df = pd.get_dummies(self.df)

    def transform(self, one_hot=False):
        if self.selected_features:
            self.filter_variables()
        if self.categorical_features:
            self.convert_categorical()
        if self.numerical_features:
            self.cast_numerical()
        if one_hot:
            self.one_hot_encoding()
        y = np.array(self.df[['Y']]).reshape(-1,)
        X = np.array(self.df.drop(['Y'], axis=1))
        return X, y


## Tests
if __name__ == "__main__":
    print(os.path.dirname(os.getcwd()), os.path.dirname('.'))
    data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "CreditTraining.csv")
    fe = FeatEng(pd.read_csv(data_path, decimal=','))

    # Define features to be selected
    fe.transform()
    print(fe.df.dtypes)
    fe.df.to_csv('../data/Credit.csv', index=False)
    print(fe.df.head())
