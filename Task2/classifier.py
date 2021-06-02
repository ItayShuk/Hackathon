import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}

def predict(X):
    pass

def send_police_cars(X):
    pass

def load(path):
    df = pd.read_csv(path)
    print(df.shape)
    df = df.dropna()
    date_copy = df["Date"].apply(lambda x: x[:10])
    # month_df = df["Date"].apply(lambda x: x[0:2])
    date_df = date_copy.apply(lambda x: datetime.date(int(x[7:]), int(x[0:2]), int(x[3:5])).weekday())
    date_df= date_df.apply(lambda x: "weekend" if 5 <= x <= 6 else "weekday")
    day_dummies = pd.get_dummies(date_df)
    # month_dummies = pd.get_dummies(month_df, prefix="month")
    df = df.join(day_dummies)
    # df = df.join(month_dummies)
    print(df)




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
    dummies = pd.get_dummies(df[''])
    print(df)



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

def main():
    # creat_files(r"Dataset_crimes.csv")
    check_data_distribution()


if __name__ == '__main__':
    main()

def creat_files(path):
    df = pd.read_csv(path)
    train = df.sample(frac=0.70)
    df = df.drop(train.index)
    test = df.sample(frac=0.10)
    valid = df.drop(test.index)
    test.to_csv("test.csv")
    train.to_csv("train.csv")
    valid.to_csv("validation.csv")