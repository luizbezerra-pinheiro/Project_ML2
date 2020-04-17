import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DataAnalysis:
    def __init__(self, df, file_name):
        self.file_name = file_name
        self.df = df

        # Methods
        self.categorical_analysis()
        self.numerical_analysis()


    def file_path(self, figure_name):
        return os.path.join(os.path.dirname(os.getcwd()), "data_analysis", self.file_name, figure_name)

    def categorical_analysis(self):
        pass

    def numerical_analysis(self):
        dtyp = self.df.dtypes
        list_numerical = list(dtyp[dtyp == "Int64"].index) + list(dtyp[dtyp == "float64"].index)
        list_numerical.remove("Y")

        for col in list_numerical:
            sns.violinplot(x="Y", y=col, data=self.df)
            plt.savefig(self.file_path(col + "_violin"))
            plt.clf()

            sns.catplot(x="Y", y=col, data=self.df)
            plt.savefig(self.file_path(col + "_catplot"))
            plt.clf()

        # print(self.df.shape)
        # aux = self.df[self.df["months_closed"] >= 10]
        # aux2 = self.df[self.df["exist_closed"] == 0]
        # print(self.df["Y"].value_counts())
        # print(aux["Y"].value_counts())
        # print(aux2["Y"].value_counts())
        # print(aux2.head(10))
        # print()
        # plt.figure()
        # # plt.scatter(aux["months_closed"], np.zeros(aux.shape[0]), c=aux["Y"])
        # plt.hist(aux[aux["Y"]==0]["months_closed"])
        # plt.hist(aux[aux["Y"] == 1]["months_closed"])
        # plt.legend()
        # plt.show()
        #
        # sns.distplot(aux[aux["Y"]==1]["months_closed"])
        # sns.distplot(aux[aux["Y"] == 0]["months_closed"])
        # plt.show()