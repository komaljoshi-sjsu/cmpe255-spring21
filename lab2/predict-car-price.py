import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#import seaborn as sns
from matplotlib import pyplot as plt
class CarPrice:
 
    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        self.reg = LinearRegression()
        self.base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):

        n = len(self.df)
        n_test = int(0.2*n)
        n_train = n - n_test

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        self.df_train = df_shuffled.iloc[:n_train].copy()
        self.df_test = df_shuffled.iloc[n_train:].copy()

        self.y_train = pd.DataFrame(np.log1p(self.df_train.msrp.values))
        self.y_test = pd.DataFrame(np.log1p(self.df_test.msrp.values)).fillna(0)

        #find correlation to get best features

        cor = self.df_train.corr().abs()['msrp']
        cor = cor.sort_values(ascending=False)
        #get top 5 correlation
        top = 5
        new_cor = cor.drop(labels = ['msrp']).head(top)
        self.cols = [None]*top
        i = 0
        for index, value in new_cor.items():
            self.cols[i] = index
            i = i+1

        del self.df_train['msrp']
        del self.df_test['msrp']

    def rmse(self, y_pred):
        mse = mean_squared_error(self.y_test,y_pred)
        return np.sqrt(mse)

    def prepare_X(self):
        x =  self.df_train[self.cols].copy()
        x.fillna(0, inplace=True)
        print('\n\n****preparing training set*****\n')
        return x

    def linear_regression(self, X):
        # ones = np.ones(X.shape[0]) 
        # X = np.column_stack([ones, X])
        # XTX = X.T.dot(X)
        # XTX_inv = np.linalg.inv(XTX)
        # w = XTX_inv.dot(X.T).dot(self.y_train)
        print('****starting regress*****')
        self.reg.fit(X,self.y_train)

    def predict(self):
        x = self.df_test[self.cols].copy()
        x.fillna(0, inplace=True)
        return self.reg.predict(x)  

    def display(self,y,error):
        df_result = pd.DataFrame(self.df_test[self.base])
        df_result['predicted_result'] = y
        df_result['actual_result'] = self.y_test
        print('\n**********Displaying Results**********\n')
        print(f'RMSE: {error}')
        print(df_result.head(5))


def test():
    carPrice = CarPrice()
    carPrice.trim()
    carPrice.validate()
    x = carPrice.prepare_X()
    carPrice.linear_regression(x) 
    y_pred = carPrice.predict()
    
    error = carPrice.rmse(y_pred)
    
    carPrice.display(y_pred,error)

if __name__ == "__main__":
    # execute only if run as a script
    test()