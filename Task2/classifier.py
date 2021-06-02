import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}

def predict(X):
    pass

def send_police_cars(X):
    pass

def load(path):
    pass
    # df = pd.read_csv(path)
    # print(df.shape)
    # df = df.dropna()
    # train = df.sample(frac=0.70)
    # df = df.drop(train.index)
    # test = df.sample(frac=0.10)
    # validation = df.drop(test.index)
    #
    # train.to_csv("train.csv")
    # test.to_csv("test.csv")
    # validation.to_csv("validation.csv")


def parser1(path):
    """
    for the primary task
    :param path:
    :return:
    """
    df = pd.read_csv(path)
    del df["ID"]
    del df["Unnamed: 0"]
    del df["Unnamed: 0.1"]
    del df["Case Number"]
    del df["Year"]
    del df["Updated On"]
    del df["IUCR"]
    del df["FBI Code"]
    del df["Description"]

    print(df)



    # df['date'] = df['date'].astype("str")
    # df['date'] = df['date'].str.slice(stop=8)
    # df['date'] = df['date'].astype("int")


def parser2(path):
    """
    for the secondary task
    :param path:
    :return:
    """
    df = pd.read_csv(path)
    del df["ID"]
    del df["Unnamed: 0"]
    del df["Unnamed: 0.1"]
    del df["Case Number"]
    del df["Year"]
    del df["Updated On"]
    print(df)

    return



def main():
    load(r"Dataset_crimes.csv")

def check_data_distribution():
    df_train = pd.read_csv("train.csv")
    df_validation = pd.read_csv("validation.csv")
    df_test = pd.read_csv("test.csv")
    df_real = pd.read_csv("Dataset_crimes.csv")
    print(" REAL CSV")
    print(df_real['Primary Type'].value_counts(normalize=True)* 100)
    print("TRAIN DATA")
    print(df_train['Primary Type'].value_counts(normalize=True)* 100)
    print("VALIDATION DATA")
    print(df_validation['Primary Type'].value_counts(normalize=True)* 100)
    print("TEST DATA")
    print(df_test['Primary Type'].value_counts(normalize=True) * 100)



if __name__ == '__main__':
    main()