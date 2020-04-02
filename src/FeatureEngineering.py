import os

import pandas as pd
from copy import copy
import numpy as np


class FeatEng:
    def __init__(self, df):
        self.df = df
        self.categorical_variables = ['Customer_Type', 'Educational_Level', 'Marital_Status',
                                     'P_Client', 'Prod_Category', 'Prod_Sub_Category',
                                     'Source', 'Type_Of_Residence']
        self.datetime_variables = ['BirthDate', 'Customer_Open_Date', 'Prod_Decision_Date', 'Prod_Closed_Date']


    def transform(self, df):
        self.convert_categorical(df)
        df = df[["Years_At_Residence", 'Net_Annual_Income', 'Y']]
        df = df[~df["Net_Annual_Income"].isna()]
        df["Net_Annual_Income"] = df["Net_Annual_Income"].str.replace(",", ".").astype("float64")
        # print("oi")
        # print(df[df["Net_Annual_Income"] == "14,4"])
        # # df["Net_Annual_Income"].str.replace(",", ".")
        # print(df["Net_Annual_Income"].str.replace(",", "."))
        return np.array(df[["Years_At_Residence", 'Net_Annual_Income']]), np.array(df["Y"])


    def convert_categorical(self, df):
        # Use LabelEncoder from sklearn to convert the categorical variables into numerical
        for column in self.categorical_variables:
            # converting type of columns to 'category'
            df[column] = df[column].astype('category')
            # Assigning numerical values and storing in another column
            df[column] = df[column].cat.codes


## Tests
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "CreditTraining.csv")
    df = pd.read_csv(data_path)

    df_fe = copy(df)
    myFE = FeatEng()
    X, y = myFE.transform(df_fe)
    print(df.head())
    print(X[:5])
