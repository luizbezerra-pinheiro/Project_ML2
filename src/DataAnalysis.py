import os
import seaborn as sns
import matplotlib.pyplot as plt


class DataAnalysis:
    def __init__(self, df, file_name):
        self.file_name = file_name
        self.df = df

        # Methods
        print("1111111111")
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