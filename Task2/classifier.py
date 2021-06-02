import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}

def predict(X):
    pass

def send_police_cars(X):
    pass

def load(path):
    df = pd.read_csv(path)
    print(df.shape)
    df = df.dropna()

    # df['date'] = df['date'].astype("str")
    # df['date'] = df['date'].str.slice(stop=8)
    # df['date'] = df['date'].astype("int")




def main():
    load(r"Dataset_crimes.csv")



if __name__ == '__main__':
    main()