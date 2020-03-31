import os

import pandas as pd
from copy import copy


def convert_categorical(df, categorical_variables):
    # Use LabelEncoder from sklearn to convert the categorical variables into numerical
    for column in categorical_variables:
        # converting type of columns to 'category'
        df[column] = df[column].astype('category')
        # Assigning numerical values and storing in another column
        df[column] = df[column].cat.codes


def transform(df, categorical_variables):
    convert_categorical(df, categorical_variables)


## Tests
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "CreditTraining.csv")
    df = pd.read_csv(data_path)

    # Define categorical_variables to be converted
    categorical_variables = ['Customer_Type', 'Educational_Level', 'Marital_Status',
                             'P_Client', 'Prod_Category', 'Prod_Sub_Category',
                             'Source', 'Type_Of_Residence']

    df_fe = copy(df)
    transform(df_fe, categorical_variables)
    print(df.head())
    print(df_fe.head())
