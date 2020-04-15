from src.GetData import GetData
from src.OurModel import OurModel

from IPython.core.display import display
import datetime as dt

import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
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
    def __init__(self, one_hot=True, scaler=True, type_analysis=0, type_select=2):
        # Type of Variables
        self.categorical_features_initial = ['Customer_Type', 'Educational_Level', 'Marital_Status',
                                             'P_Client', 'Prod_Category', 'Prod_Sub_Category',
                                             'Source', 'Type_Of_Residence']
        self.categorical_features_final = self.categorical_features_initial + ["exist_closed"]
        self.numerical_features_initial = ['Net_Annual_Income', 'Number_Of_Dependant', 'Years_At_Residence',
                                           'Years_At_Business', 'Nb_Of_Products']
        self.numerical_features_final = self.numerical_features_initial + ["age", "months_customer", "months_decision",
                                                                           "months_closed"]
        self.datetime_variables = ['BirthDate', 'Customer_Open_Date', 'Prod_Decision_Date', 'Prod_Closed_Date']

        # OneHotEncoder
        self.one_hot = one_hot  # We use FAMD or not. If not, we have to do one hot encoder.
        self.encoders = None

        # Scaler
        self.scaler = scaler
        self.scaler_Robust = None

        # Dimensionality Reduction
        # 0 (nothing), 1 (pca_mca), 2 (famd)
        self.type_analysis = type_analysis
        self.my_pca = None
        self.my_mca = None
        self.my_FAMD = None

        # Selected features
        # 0 (Nothing), 1 (best componenents dimensionality reduction), 2 (feature_importance of random forest)
        self.type_select = type_select
        self.best_cols_rf = None

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
        return self.transform(df)


    def transform(self, df):
        # Treating categorical, numerical and datetime variables
        df = self.missing_values(df)
        df = self.grouping_categorical(df)
        df = self.casting_categorical_numerical(df)
        df = self.preprocess_datetime(df)
        df = self.filter_variables(df)

        # One hot encoder
        if self.one_hot is True:
            df = self.labelEncoder(df)

        # Scaler
        if self.scaler is True:
            df = self.numerical_scaling(df)

        # Type of the dimension reduction
        if self.type_analysis == 1:
            df = self.dimension_reduction_pca_mca(df)
        if self.type_analysis == 2:
            df = self.dimension_reduction_famd(df)

        # Feature Selection
        if self.type_select == 2:
            if self.best_cols_rf is None:
                self.best_cols_rf = self.feature_selection_rf(df)
            df = df[self.best_cols_rf]

        return df


    def grouping_categorical(self, df):
        df = df.replace(["Secondary or Less", "Diploma"], ["University", "Master/PhD"])
        df = df.replace(["Separated", "Widowed", "Divorced"], "Single")
        df = df.replace(["A", "F", "I", "J", "M"], "Other")
        df = df.replace("Company", "Owned")

        return df


    def filter_variables(self, df):
        drop_variables = ["Id_Customer"]
        return df.drop(drop_variables, axis=1)


    def casting_categorical_numerical(self, df):
        # Convert the type of the categorical variables
        df[self.categorical_features_initial] = df[self.categorical_features_initial].astype("category")
        # Float values
        df[self.numerical_features_initial] = df[self.numerical_features_initial].astype('float64')

        return df


    def labelEncoder(self, df):
        # Fitting OneHotEncoder, if one_hot is True
        if self.encoders is None:
            self.encoders = OneHotEncoder(handle_unknown='ignore')
            self.encoders.fit(df[self.categorical_features_initial])

        cols = self.encoders.get_feature_names(self.categorical_features_initial)
        aux = pd.DataFrame(self.encoders.transform(df[self.categorical_features_initial]).toarray(), columns=cols)
        aux[cols] = aux[cols].astype('category')
        df = df.drop(self.categorical_features_initial, axis=1)
        df = df.join(aux)

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
        cols = self.numerical_features_final
        X_aux = np.array(df[cols])
        feature_name = df[cols].columns
        type_dic = dict(df[cols].dtypes)

        # If scaler is 0, it means it is df_train. So we fit the scaler
        if self.scaler_Robust is None:
            self.scaler_Robust = RobustScaler()
            self.scaler_Robust.fit(X_aux)

        X_aux_scaled = self.scaler_Robust.transform(X_aux)

        df[feature_name] = X_aux_scaled
        df = df.astype(type_dic)

        return df


    def dimension_reduction_pca_mca(self,df):
        ### MCA - Categorical features
        print("--------------------------")
        print(df.head())
        cat_cols = []
        if self.one_hot is True:
            cat_cols = self.encoders.get_feature_names(self.categorical_features_initial) + ["exist_closed"]
        else:
            cat_cols = self.categorical_features_final
        print(cat_cols)
        # If my_mca is None, it means it is df_train. So we fit my_mca
        n_comp = len(cat_cols)
        if self.my_mca is None:
            self.my_mca = prince.MCA(n_components=n_comp)
            self.my_mca.fit(df[cat_cols])

        cols = []
        for i in range(n_comp):
            cols += ["MCA_"+str(i+1)]

        print(df[cat_cols].shape)
        print(self.my_mca)
        aux_mca = self.my_mca.transform(df[cat_cols])
        aux_mca.columns = cols
        df = df.drop(cat_cols, axis=1)
        df = df.join(aux_mca)

        ## PCA - Numerical features

        # If my_pca is 0, it means it is df_train. So we fit my_pca
        num_cols = self.numerical_features_final
        n_comp = len(num_cols)
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


    def dimension_reduction_famd(self, df):
        n_comp = df.shape[1]
        col = []
        for i in range(n_comp):
            col.append("FAMD_" + str(i + 1))

        if self.my_FAMD is None:
            self.my_FAMD = prince.FAMD(n_components=n_comp,
                                       n_iter=10,
                                       copy=True,
                                       check_input=True,
                                       engine='auto',
                                       random_state=42)
            self.my_FAMD = self.my_FAMD.fit(df.drop("Y", axis=1))
            print(self.my_FAMD.explained_inertia_)
        else:
            print("NAOOOOOOOO")
            print(df.drop("Y", axis=1).shape)
            print(self.my_FAMD)
        aux = self.my_FAMD.transform(df.drop("Y", axis=1))
        aux.columns = col
        aux = aux.join(df["Y"])
        df = aux

        print("Depois FAMD:")
        print(df.head())

        plt.figure(figsize=(12, 12))

        plt.scatter(df[df["Y"]==0]["FAMD_3"], df[df["Y"]==0]["FAMD_4"], color='red', alpha=0.5, label='0')
        plt.scatter(df[df["Y"]==1]["FAMD_3"], df[df["Y"]==1]["FAMD_4"], color='blue', alpha=0.5, label='1')
        plt.title("FAMD")
        plt.ylabel('Les coordonnees de Y')
        plt.xlabel('Les coordonnees de X')
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 12))
        sns.distplot(df[df["Y"]==1]["FAMD_3"])
        sns.distplot(df[df["Y"]==0]["FAMD_3"])
        plt.show()

        return df


    def feature_selection_rf(self, df):
        aux = df.drop("Y", axis=1)

        rf = OurModel().models[0]
        rf.fit(aux, df["Y"])
        feat_imp = [(col, v) for col, v in zip(aux.columns, rf.feature_importances_)]
        feat_imp = sorted(feat_imp, key=lambda x: x[1], reverse=True)

        best_cols = [el[0] for el in feat_imp]

        return best_cols[:11] + ["Y"]

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