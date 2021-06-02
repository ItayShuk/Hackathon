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
    # print(df.shape)
    df = df.dropna()

def main():
    load()



if __name__ == '__main__':
    main()