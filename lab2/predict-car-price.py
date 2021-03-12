import pandas as pd
import numpy as np

#import seaborn as sns
from matplotlib import pyplot as plt
class CarPrice:
 
    def __init__(self):
        self.df = pd.read_csv('data/data.csv')
        self.base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
        print(f'${len(self.df)} lines loaded')

    def trim(self):
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        string_columns = list(self.df.dtypes[self.df.dtypes == 'object'].index)
        for col in string_columns:
            self.df[col] = self.df[col].str.lower().str.replace(' ', '_')

    def validate(self):

        n = len(self.df)
        n_val = int(0.2 * n)
        n_test = int(0.2 * n)
        n_train = n - (n_val + n_test)

        idx = np.arange(n)
        np.random.shuffle(idx)

        df_shuffled = self.df.iloc[idx]

        self.df_train = df_shuffled.iloc[:n_train].copy()
        self.df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
        self.df_test = df_shuffled.iloc[n_train+n_val:].copy()

        self.y_train = np.log1p(self.df_train.msrp.values)
        self.y_val = np.log1p(self.df_val.msrp.values)
        self.y_test = np.log1p(self.df_test.msrp.values)
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
        del self.df_val['msrp']

    def get_x_train(self):
        return self.df_train

    def get_y_train(self):
        return self.y_train

    def get_x_val(self):
        return self.df_val
    
    def get_y_val(self):
        return self.y_val

    def rmse(self,y_pred,y):
        error = y_pred - y
        mse = (error ** 2).mean()
        return np.sqrt(mse)

    def prepare_X(self,input):
        #use top 5 correlation to train data
        x =  input[self.cols].copy()
        x =  input[self.cols].copy()
        x.fillna(0, inplace=True)
        return x.values

    def linear_regression(self,X):
        #use transpose to find weight
        ones = np.ones(X.shape[0])
        X = np.column_stack([ones, X])

        XTX = X.T.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        w = XTX_inv.dot(X.T).dot(self.y_train)
        
        return w[0], w[1:] 

    def display(self,x,y_predicted,y_actual,error,set_type):
        df_result = pd.DataFrame(x[self.base])

        #converting values back to normal
        df_result['predicted_result'] = np.expm1(y_predicted)
        df_result['actual_result'] = np.expm1(y_actual)
        print('\n**********Displaying Results for '+set_type+' set**********\n')
        print(f'RMSE: {error}')
        print(df_result.head(5))


def test():
    carPrice = CarPrice()
    carPrice.trim()
    carPrice.validate()
    x = carPrice.prepare_X(carPrice.get_x_train())

    w_0, w = carPrice.linear_regression(x)

    y_pred = w_0 + x.dot(w)
    error = carPrice.rmse(y_pred,carPrice.get_y_train())
    carPrice.display(carPrice.get_x_train(),y_pred,carPrice.get_y_train(),error,'Training')

    x_val = carPrice.prepare_X(carPrice.get_x_val())
    y_val_pred = w_0 + x_val.dot(w)
    error_val = carPrice.rmse(y_val_pred,carPrice.get_y_val())
    carPrice.display(carPrice.get_x_val(),y_val_pred,carPrice.get_y_val(),error_val,'Validation')

if __name__ == "__main__":
    # execute only if run as a script
    test()