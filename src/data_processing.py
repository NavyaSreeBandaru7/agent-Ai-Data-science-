import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path='data/iris.csv'):
    return pd.read_csv(path)

def preprocess(df):
    X = df.drop('species', axis=1)
    y = df['species']
    return train_test_split(X, y, test_size=0.2, random_state=42)
