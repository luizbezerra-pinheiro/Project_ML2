from src.GetData import GetData

from IPython.core.display import display
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 100)


"""
Class responsible for preprocessing our Data
"""


class FeatEng:
    def __init__(self, selected_feat=False):
        # Selected Variables
        self.selected_feat = selected_feat
        self.selected_features = ["Y", "Net_Annual_Income", "Years_At_Business",
                                  "Years_At_Residence", "Number_Of_Dependant"]

        # Type of Variables
        self.categorical_features = ['Customer_Type', 'Educational_Level', 'Marital_Status',
                                     'P_Client', 'Prod_Category', 'Prod_Sub_Category',
                                     'Source', 'Type_Of_Residence']
        self.numerical_features = {'float64': ['Net_Annual_Income'],
                                   'Int64': ['Number_Of_Dependant', 'Years_At_Residence',
                                             'Years_At_Business', 'Nb_Of_Products']}
        self.datetime_variables = ['BirthDate', 'Customer_Open_Date', 'Prod_Decision_Date', 'Prod_Closed_Date']

        # OneHotEncoder
        self.one_hot = False  # We don't have yet the encoder
        self.encoders = 0


    def fit(self, df):
        # Fitting OneHotEncoder
        self.encoders = OneHotEncoder(handle_unknown='ignore')
        self.encoders.fit(df[self.categorical_features])
        self.one_hot = True  # We have now the encoders


    def transform(self, df):
        # Treating categorical and numerical variables
        df = self.convert_categorical(df)
        df = self.cast_numerical(df)

        # select columns, if in case
        if self.selected_feat:
            df = self.filter_variables(df)

        # Data and result
        y = np.array(df[['Y']]).reshape(-1, )
        X = np.array(df.drop(['Y'], axis=1))

        return X, y


    def filter_variables(self, df):
        return df[self.selected_features].fillna(0)


    def convert_categorical(self, df):
        # Use LabelEncoder from sklearn to convert the categorical variables into numerical
        if self.one_hot is False:
            raise Exception("There is no Label Encoder fitted.")

        aux = pd.DataFrame(self.encoders.transform(df[self.categorical_features]).toarray(),
                           columns=self.encoders.get_feature_names(self.categorical_features))

        df = df.drop(self.categorical_features, axis=1)
        df = df.join(aux)

        return df


    def cast_numerical(self, df):
        # List of float and Int columns
        selected_numerical_float = list(set(self.numerical_features['float64']))
        selected_numerical_int = list(set(self.numerical_features['Int64'])) + list(self.encoders.get_feature_names(self.categorical_features))

        # Float values
        df[selected_numerical_float] = df[selected_numerical_float].astype('float64')

        # Integer values
        # Int allows NaN values (int doesn't allow it)
        df[selected_numerical_int] = df[selected_numerical_int].astype('Int64')

        return df


## Tests
if __name__ == "__main__":
    df_train, df_test = GetData().get()

    fe = FeatEng()
    fe.fit(df_train)

    # Tests

    # Original dataset
    print("\n### Original dataset")
    display(df_train.head())

    # Testing convert_categorical function
    print("\n### Categorical features")
    aux = fe.convert_categorical(df_train)
    print("Shape:", aux.shape)
    display(aux.head())

    # Testing cast_numerical function
    print("\n### Numerical variables")
    aux = fe.cast_numerical(aux)
    print(aux.dtypes)

    # Transform
    print("\n### Transform function")
    X_train, y_train = fe.transform(df_train)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    X_test, y_test = fe.transform(df_test)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # fe.df.to_csv('../data/Credit_FeatEng.csv', index=False)