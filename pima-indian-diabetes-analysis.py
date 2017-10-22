import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates, andrews_curves, scatter_matrix




class Visualization():

    def __init__(self, df):
        self.df = df

    def normalize(self):
        # df_temp = self.df.drop(labels='diabetes?', axis=1)
        df_norm = (self.df - self.df.mean()) / (self.df.max() - self.df.min())
        return df_norm

    def descriptives(self):
        # Drop class label column because we don't need it for dercriptive statistics
        df_temp = self.df.drop(labels='diabetes?', axis=1)
        return df_temp.describe()


    def bar_plot(self, df):
        df_0 = df[df['diabetes?'] == 0]
        df_1 = df[df['diabetes?'] == 1]
        mean_0 = df_0.mean()
        mean_1 = df_1.mean()
        list_0 = []
        list_1 = []
        for col in df.columns:
            list_0.append(mean_0[col])
            list_1.append(mean_1[col])

        index = np.arange(len(df.columns))
        bar_width = 0.35
        opacity = 0.8

        plt.bar(index, list_0, bar_width,
                alpha=opacity,
                color='b',
                label='Diabetes (-ve)')

        plt.bar(index + bar_width, list_1, bar_width,
                alpha=opacity,
                color='g',
                label='Diabebtes (+ve)')

        plt.title('Pima Indians Diabetes Dataset')
        plt.xticks(index + bar_width, tuple(df.columns))
        plt.legend()

        plt.tight_layout()
        plt.show()

    def box_plot(self, df, cols):
        data = np.array(df.values.tolist())
        plt.boxplot(data, labels=cols, showmeans=True)
        plt.show()

    def scatter_matrix(self, df):
        colors = list('r' if i==1 else 'b' for i in self.df['diabetes?'])
        scatter_matrix(df, color=colors)
        plt.show()

    def stacked_histogram(self, df, cols):
        fig, axes = plt.subplots(3, 3, sharey=True) # Because I have 9 variables in the dataset
        df_1 = df[df['diabetes?'] == 1]
        df_2 = df[df['diabetes?'] == 0]
        col_index = 0
        for row in axes:
            for col in row:
                col.hist([df_2[cols[col_index]], df_1[cols[col_index]]], bins=10, stacked=True, color=['b', 'g'])
                col.set_ylabel(cols[col_index])
                col_index += 1
        plt.show()


if __name__ == '__main__':
    filename = 'D:/blog/random_datasets/pima-indians-diabetes.csv'
    df = pd.read_csv(filename, sep=',', encoding='utf-8', header=None)
    cols = df.columns = ['n_pregnant', 'glu_conc', 'bp', 'tst', 'insulin', 'bmi', 'dpf', 'age', 'diabetes?']
    metainfo = {'n_pregnant': 'Number of times pregnant',
                     'glu_conc': 'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
                     'bp': 'Diastolic blood pressure (mm Hg)',
                     'tst': 'Triceps skin fold thickness (mm)',
                     'insulin': '2-Hour serum insulin (mu U/ml)',
                     'bmi': 'Body mass index (weight in kg/(height in m)^2)',
                     'dpf': 'Diabetes pedigree function',
                     'age': 'Age (years)',
                     'diabetes?': 'Class variable (0 or 1)'}
    obj = Visualization(df)
    descriptive_stats = obj.descriptives()
    normalized_data = obj.normalize()
    normalized_stats = obj.normalize().describe()
    print(descriptive_stats)
    print(normalized_stats)
    # obj.bar_plot(df)
    # obj.box_plot(df, cols)
    # obj.box_plot(normalized_data, cols)
    obj.scatter_matrix(normalized_data)
    # obj.stacked_histogram(df, cols)