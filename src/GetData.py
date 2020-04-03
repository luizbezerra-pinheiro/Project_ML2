import os
from copy import copy
import pandas as pd
from sklearn.model_selection import train_test_split


"""
Class responsible for giving the database
"""


class GetData:
    def __init__(self):
        self.data_path = os.path.join(os.path.dirname(os.getcwd()), "data", "CreditTraining.csv")
        self.df = copy(pd.read_csv(self.data_path))

    """
    Method called by the user to get the data split and organized 
    
    Divide the database in train and test
    We will create a model with the train. 
    We will use the df_test at the final to see if the model works well
    """

    def get(self):
        df = self.df

        # Splitting the database
        X_train, X_test, y_train, y_test = train_test_split(df.drop("Y", axis=1), df["Y"], test_size=0.1,
                                                            random_state=7)
        df_train = pd.concat([y_train, X_train], axis=1).reset_index(drop=True)
        df_test = pd.concat([y_test, X_test], axis=1).reset_index(drop=True)

        return df_train, df_test


"""
Testing the functions of the class
"""
if __name__ == "__main__":
    df_train, df_test = GetData().get()

    print("\n### Analysing each dataset")

    print("\n-- Train")
    print("Shape:", df_train.shape)
    print("Distribution of class")
    print(df_train["Y"].value_counts() / df_train.shape[0])

    print("\n-- Test")
    print("Shape:", df_test.shape)
    print("Distribution of class")
    print(df_test["Y"].value_counts() / df_test.shape[0])
