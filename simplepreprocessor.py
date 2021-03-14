import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class simplePreprocessor():

    def convert(self, df):

        self.X = df.iloc[:, :-1]
        self.Y = df.iloc[:, -1]

        X = np.asarray(self.X)
        Y = np.asarray(self.Y).reshape(-1, 1)

        return X, Y
    
    def standardize_data(self, X):

        X = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
        
        return X

    def split_data(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

        return X_train, X_test, y_train, y_test

    def preprocess(self, df):

        X, y = self.convert(df)
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        X_train = self.standardize_data(X_train)
        X_test = self.standardize_data(X_test)

        return X_train.T, X_test.T, y_train.T, y_test.T


     
