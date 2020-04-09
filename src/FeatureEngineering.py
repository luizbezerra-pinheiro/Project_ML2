from src.GetData import GetData

from IPython.core.display import display
import datetime as dt

import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import prince

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 100)


"""
Class responsible for preprocessing our Data
"""


class FeatEng:
    def __init__(self):
        # Type of Variables
        self.categorical_features = ['Customer_Type', 'Educational_Level', 'Marital_Status',
                                     'P_Client', 'Prod_Category', 'Prod_Sub_Category',
                                     'Source', 'Type_Of_Residence']
        self.numerical_features = ['Net_Annual_Income', 'Number_Of_Dependant', 'Years_At_Residence',
                                   'Years_At_Business', 'Nb_Of_Products']
        # self.numerical_features = {'float64': ['Net_Annual_Income'],
        #                            'Int64': ['Number_Of_Dependant', 'Years_At_Residence',
        #                                      'Years_At_Business', 'Nb_Of_Products']}
        self.datetime_variables = ['BirthDate', 'Customer_Open_Date', 'Prod_Decision_Date', 'Prod_Closed_Date']

        # OneHotEncoder
        self.one_hot = False  # We don't have yet the encoder
        self.encoders = 0

        # Scaler
        self.scaler_Robust = 0

        # Dimensionality Reduction
        self.my_pca = None
        self.my_mca = None

        # Assumption
        self.today = dt.datetime(2014, 1, 1)
        self.mode = {"Number_Of_Dependant": -1,
                     "Net_Annual_Income": -1,
                     "Years_At_Business": -1
                     }


    def fit_transform(self, df):
        # Mode and mean for missing values
        for d in ["Number_Of_Dependant", "Net_Annual_Income", "Years_At_Business"]:
            self.mode[d] = float(df[d].mode())

        ## Não vai ser necessário fazer OneHotEncoder pois o MCA faz isso no algorithmo

        # Fitting OneHotEncoder
        self.encoders = OneHotEncoder(handle_unknown='ignore')
        self.encoders.fit(df[self.categorical_features])
        self.one_hot = True  # We have now the encoders

        return self.transform(df)


    def transform(self, df):
        # Treating categorical and numerical variables
        df = self.convert_categorical(df)
        df = self.cast_numerical(df)
        df = self.preprocess_datetime(df)
        df = self.missing_values(df)

        df = self.filter_variables(df)

        df = self.numerical_scaling(df)
        # df = self.dimension_reduction(df)

        # # Data and result
        # y = np.array(df[['Y']]).reshape(-1, )
        # X = np.array(df.drop(['Y'], axis=1))
        #
        # return X, y, df.drop(['Y'], axis=1).columns

        return df


    def filter_variables(self, df):
        drop_variables = ["Id_Customer"]
        return df.drop(drop_variables, axis=1)


    def convert_categorical(self, df):
        # Use LabelEncoder from sklearn to convert the categorical variables into numerical
        if self.one_hot is False:
            raise Exception("There is no Label Encoder fitted.")

        cols = self.encoders.get_feature_names(self.categorical_features)
        aux = pd.DataFrame(self.encoders.transform(df[self.categorical_features]).toarray(), columns=cols)
        aux[cols] = aux[cols].astype('category')

        df = df.drop(self.categorical_features, axis=1)
        df = df.join(aux)

        return df


    def cast_numerical(self, df):
        # List of float and Int columns
        selected_numerical_float = self.numerical_features
        # selected_numerical_float = list(set(self.numerical_features['float64'])) + list(set(self.numerical_features['Int64']))
        # selected_numerical_int = list(set(self.numerical_features['Int64']))

        # Float values
        df[selected_numerical_float] = df[selected_numerical_float].astype('float64')

        # Integer values
        # Int allows NaN values (int doesn't allow it)
        # df[selected_numerical_int] = df[selected_numerical_int].astype('Int64')

        return df


    def preprocess_datetime(self, df):
        # Changing the type of the column
        for d in self.datetime_variables:
            df[d] = pd.to_datetime(df[d])

        # Age from BirthDate
        df["age"] = df["BirthDate"].apply(lambda x: (self.today - x).days/365).astype('float64')
        df = df.drop("BirthDate", axis=1)

        # Months from Customer_Open_Date
        df["months_customer"] = df["Customer_Open_Date"].apply(lambda x: (self.today - x).days/30).astype('float64')
        df = df.drop("Customer_Open_Date", axis=1)

        # Months from Prod_Decision_Date
        df["months_decision"] = df["Prod_Decision_Date"].apply(lambda x: (self.today - x).days / 30).astype("float64")
        df = df.drop("Prod_Decision_Date", axis=1)

        # Months from Prod_Closed_Date and boolean if exists Prod_Closed_Date
        df["exist_closed"] = (df["Prod_Closed_Date"].notnull() * 1).astype("category")
        df["months_closed"] = df["Prod_Closed_Date"].apply(lambda x: (self.today - x).days / 30).fillna(-1).astype('float64')

        df = df.drop("Prod_Closed_Date", axis=1)

        return df


    def missing_values(self, df):
        for d in ["Number_Of_Dependant", "Net_Annual_Income", "Years_At_Business"]:
            df[d] = df[d].fillna(self.mode[d])

        return df


    def numerical_scaling(self, df):
        cols = self.numerical_features
        X_aux = np.array(df[cols])
        feature_name = df[cols].columns
        type_dic = dict(df[cols].dtypes)

        # If scaler is 0, it means it is df_train. So we fit the scaler
        if self.scaler_Robust == 0:
            self.scaler_Robust = RobustScaler()
            self.scaler_Robust.fit(X_aux)

        X_aux_scaled = self.scaler_Robust.transform(X_aux)

        df[feature_name] = X_aux_scaled
        df = df.astype(type_dic)

        return df


    def dimension_reduction(self, df):
        # ### MCA - Categorical features
        # print("--------------------------")
        # print(df.head())
        # cat_cols = self.categorical_features + ["exist_closed"]
        # print(cat_cols)
        # df[cat_cols] = df[cat_cols].astype("category")
        # # If my_mca is None, it means it is df_train. So we fit my_mca
        # n_comp = 4
        # if self.my_mca is None:
        #     self.my_mca = prince.MCA(n_components=n_comp)
        #     self.my_mca.fit(df[cat_cols])
        #
        # cols = []
        # for i in range(n_comp):
        #     cols += ["MCA_"+str(i+1)]
        #
        # print(df[cat_cols].shape)
        # print(self.my_mca)
        # aux_mca = self.my_mca.transform(df[cat_cols])
        # aux_mca.columns = cols
        # df = df.drop(cat_cols, axis=1)
        # df = df.join(aux_mca)

        ### PCA - Numerical features

        # If my_pca is 0, it means it is df_train. So we fit my_pca
        n_comp = 2
        num_cols = self.numerical_features + ["age", "months_customer", "months_decision", "months_closed"]
        print(num_cols)
        if self.my_pca is None:
            self.my_pca = PCA(n_components=n_comp)
            self.my_pca.fit(df[num_cols])

        cols = []
        for i in range(n_comp):
            cols += ["PCA_" + str(i + 1)]

        print(df.head())
        print(num_cols)
        aux_pca = pd.DataFrame(self.my_pca.transform(df[num_cols]), columns=cols)
        df = df.drop(num_cols, axis=1)
        df = df.join(aux_pca)
        print(self.my_pca.explained_variance_ratio_)
        print(df.head())





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

    # Testing preprocess_datetime method
    print("\n### Datetime variables")
    aux = fe.preprocess_datetime(aux)
    print(aux.head(10))

    # Missing values
    print("\n### Missing values")
    aux = fe.missing_values(aux)
    display(aux.describe())

    # Transform
    print("\n### Transform function")
    X_train, y_train = fe.transform(df_train)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    X_test, y_test = fe.transform(df_test)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)


    # fe.df.to_csv('../data/Credit_FeatEng.csv', index=False)