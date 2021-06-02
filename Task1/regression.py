
################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def load_data(path):
    df = pd.read_csv(path)
    print(df.shape)

    df = df.dropna()

    #####date cleaning
    df['date'] = df['date'].astype("str")
    df['date'] = df['date'].str.slice(stop=8)
    df['date'] = df['date'].astype("int")

    #####negative numbers
    pos_cols = np.arange(21)
    pos_cols = np.delete(pos_cols, 18)
    df = df[(df[df.columns[pos_cols]] >= 0).all(1)]
    # print(df.shape)

    ######sane number of bedrooms,bathrooms
    rooms = [3, 4]
    df = df[(df[df.columns[rooms]] > 0).all(1)]
    df = df[(df[df.columns[rooms]] < 10).all(1)]
    # print(df.shape)




    dummies = pd.get_dummies(df['zipcode'])
    del df['zipcode']
    del df['lat']
    del df['long']
    del df['id']
    df = df.join(dummies)
    return df

def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """
    #your code goes here...
    pass

def main():
    load_data(r"C:\Users\ITAY\Desktop\Hackhathon\Task1\movies_dataset.csv")


if __name__ == '__main__':
    main()