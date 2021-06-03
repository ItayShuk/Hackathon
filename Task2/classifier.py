import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}
crimes_dict_rev = {'BATTERY' : 0, 'THEFT': 1, 'CRIMINAL DAMAGE': 2, 'DECEPTIVE PRACTICE': 3, 'ASSAULT': 4}


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




def parser1(df):
    """
    for the primary task
    :param path:
    :return:
    """
    del df["ID"]
    del df["Unnamed: 0"]
    del df["Unnamed: 0.1"]
    del df["Case Number"]
    del df["Year"]
    del df["Updated On"]
    del df["IUCR"]
    del df["FBI Code"]
    del df["Description"]
    del df["Latitude"]
    del df["Longitude"]
    del df["Location"]
    df.replace({"Primary Type": crimes_dict_rev}, inplace=True)
    # dummies = pd.get_dummies(df[''])
    time = df["Date"].apply(lambda x: int(x[11:13]) if x[20:] == "AM" else int(x[11:13]) + 12)
    # print(time)
    del df["Date"]
    df = df.join(time)
    df.rename(columns={"Date": "Time"}, inplace=True)
    morning = df[(df['Time'] >= 6) & (df['Time'] < 14)]
    noon = df[(df['Time'] >= 14) & (df['Time'] < 22)]
    night = df[((df['Time'] >= 22) & (df['Time'] <= 24) |
                (df['Time'] >= 0) & (df['Time'] < 6))]
    return morning, noon, night

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
    test()

def area_preprocessor():
    df = pd.read_csv("train.csv")
    # print(df.shape)
    df = df.dropna()
    # print(df.shape)
    x = df["X Coordinate"].to_numpy()
    y = df["Y Coordinate"].to_numpy()
    # z = df["District"].to_numpy()
    z = df["Block"].apply(lambda x : int(x[0:3])).to_numpy()
    z = df["Beat"].to_numpy()
    plt.scatter(x, y, s=1, c=z)
    plt.show()


def split(df: pd.DataFrame):
    y = df["Primary Type"]
    X = df.drop(["Primary Type"], axis=1)
    return X.to_numpy(), y.to_numpy()


def train_trees(df: pd.DataFrame):
    df_morning, df_noon, df_night = parser1(df)
    df_morning = df_morning.drop(["Block", "Location Description", "Time"], axis=1)
    df_noon = df_noon.drop(["Block", "Location Description", "Time"], axis=1)
    df_night = df_night.drop(["Block", "Location Description", "Time"], axis=1)
    X1, y1 = split(df_morning)
    X2, y2 = split(df_noon)
    X3, y3 = split(df_night)
    class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}
    model_morning = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, class_weight=class_weights)
    model_noon = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, class_weight=class_weights)
    model_night = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, class_weight=class_weights)
    return model_morning.fit(X1, y1), model_noon.fit(X2, y2), model_night.fit(X3, y3)


def trees_predict(T1, T2, T3, X_test):
    return T1.predict(X_test)

def test():
    df = pd.read_csv("train.csv")
    df = df.dropna()
    df_train = df.sample(frac=0.8)
    df_test = df.drop(df_train.index)
    T1, T2, T3 = train_trees(df_train)
    # y_hat = trees_predict(T1, T2, T3, X_test)
    df1_test, df2_test, df3_test = parser1(df_test)
    df1_test = df1_test.drop(["Block", "Location Description", "Time"], axis=1)
    X1_test, y1_test = split(df1_test)
    # print(T1.score(X1_test, y1_test))


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

def creat_files(path):
    df = pd.read_csv(path)
    train = df.sample(frac=0.70)
    df = df.drop(train.index)
    test = df.sample(frac=0.10)
    valid = df.drop(test.index)
    test.to_csv("test.csv")
    train.to_csv("train.csv")
    valid.to_csv("validation.csv")

def rotem():
    df = pd.read_csv("train.csv")
    morning = df[(df['Date'] >= 0) and (df['Date'] < 8)]
    noon = df[(df['Date'] >= 8) and (df['Date'] < 16)]
    night = df[(df['Date'] >= 16) and (df['Date'] < 24)]

    pass

class Bagger:

    morning_df = None
    noon_df = None
    night_df = None

    morning_hat = None
    noon_hat = None
    night_hat = None

    morning_bagger = None
    noon_bagger = None
    night_bagger = None

    def __init__(self, morning_df, morning_hat, noon_df, noon_hat, night_df, night_hat):

        self.morning_df = morning_df
        self.noon_df = noon_df
        self.night_df = night_df

        self.morning_hat = morning_hat
        self.noon_hat = noon_hat
        self.night_hat = night_hat

    def train_committee(self, T, max_features, max_depth, min_samples_leaf):
        class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}

        self.morning_bagger = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, class_weight=class_weights),
                                                T, max_samples=self.morning_df.shape[0], max_features=max_features, bootstrap=True).fit(self.morning_df, self.morning_hat)

        self.noon_bagger = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, class_weight=class_weights),
                                             T,  max_samples=self.noon_df.shape[0], max_features=max_features, bootstrap=True).fit(self.noon_df, self.noon_hat)

        self.night_bagger = BaggingClassifier(DecisionTreeClassifier(max_depth=max_depth,min_samples_leaf=min_samples_leaf, class_weight=class_weights),
                                              T,  max_samples=self.night_df.shape[0], max_features=max_features, bootstrap=True).fit(self.night_df, self.night_hat)


def get_data_rotem():
    df = pd.read_csv("train.csv")
    df = df.dropna()
    df_train = df.sample(frac=0.8)
    df_test = df.drop(df_train.index)

    df_morning, df_noon, df_night = parser1(df)
    df_morning = df_morning.drop(["Block", "Location Description", "Time"], axis=1)
    df_noon = df_noon.drop(["Block", "Location Description", "Time"], axis=1)
    df_night = df_night.drop(["Block", "Location Description", "Time"], axis=1)
    X1, y1 = split(df_morning)
    X2, y2 = split(df_noon)
    X3, y3 = split(df_night)

    df1_test, df2_test, df3_test = parser1(df_test)
    df1_test = df1_test.drop(["Block", "Location Description", "Time"], axis=1)
    df2_test = df2_test.drop(["Block", "Location Description", "Time"], axis=1)
    df3_test = df3_test.drop(["Block", "Location Description", "Time"], axis=1)

    X1_test, y1_test = split(df1_test)
    X2_test, y2_test = split(df2_test)
    X3_test, y3_test = split(df3_test)



    all_features = X1.shape[1]
    T = [5, 10, 15, 20, 50, 100, 200]
    max_samples = [2000, 3000, 4000] #6360 morning, 7500 noon, 4817 night
    max_features = [3, 4, 5, 6, 7, all_features]
    max_depth = [5, 10, 15, 20, 25]
    min_samples_leaf = np.arange(5, 500, step=10)
    counter = 0
    for t in T:
        for m in max_features:
            plt.figure()
            morning_score = []
            noon_score = []
            night_score = []
            counter += 1
            for d in max_depth:

                my_bagger = Bagger(X1, y1, X2, y2, X3, y3)
                my_bagger.train_committee(t, m, d, 30)
                morning_score.append(my_bagger.morning_bagger.score(X1_test, y1_test))
                noon_score.append(my_bagger.noon_bagger.score(X2_test, y2_test))
                night_score.append(my_bagger.night_bagger.score(X3_test, y3_test))
            plt.plot(max_depth, morning_score, label="Morning bagger")
            plt.plot(max_depth, noon_score, label="Noon bagger")
            plt.plot(max_depth, night_score, label="Night bagger")
            plt.xlabel("max depth")
            plt.ylabel("score")
            plt.title("Score of baggers with committee T = " + str(t) + " and max_features = " + str(m))
            plt.legend()
            image_name = str(m) + "features" + str(counter)
            plt.savefig(image_name + "png", dpi=300, bbox_inches='tight')
            plt.show()

get_data_rotem()

