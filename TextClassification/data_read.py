import pandas as pd


def get_data():
    data = pd.read_csv(r"D:\Pytorch_Tutorial\TextClassification\data\movie_data.csv")
    data_select = data.loc[101:200]

    # data_select.to_csv(r"D:\Pytorch_Tutorial\TextClassification\data\valid_movie_data_sample.csv", index=False)
    print(data_select['sentiment'].unique())
    print(data_select.columns.to_list())


get_data()
