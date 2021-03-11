import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class Linear_Ref:
    def __init__(self) :
        print("here")
        self.final_rmse = None
        self.final_r2 = None
        self.final_df_predict = None
        self.final_df_test = None
        self.final_feature_name = None

        boston = load_boston()
        self.df_data = pd.DataFrame(boston.data,columns=boston.feature_names)
        self.df_target = pd.DataFrame(boston.target)
        self.n = len(self.df_data)
        self.n_test = int(0.2*self.n)
        self.n_train = self.n - self.n_test

        self.df_target_train = self.df_target.iloc[:self.n_train].copy()
        self.df_target_test = self.df_target.iloc[self.n_train:].copy()

        self.reg = LinearRegression()

    def split_train_test(self,data):
        df_train = data.iloc[:self.n_train].copy()
        df_test = data.iloc[self.n_train:].copy()
        return df_train,df_test

    def linear_reg(self,df_train):
        self.reg.fit(df_train,self.df_target_train)

    def predict(self,df_test):
        return pd.DataFrame(self.reg.predict(df_test))

    def rmse(self,df_predict):
        mse = mean_squared_error(self.df_target_test,df_predict)
        return np.sqrt(mse)

    def r_sq_score(self,df_predict):
        return r2_score(self.df_target_test, df_predict,multioutput='variance_weighted')


    def plot(self):
        print('Most optimum col is: '+self.final_feature_name)
        print('R-Squared Value is: '+str(self.final_r2))
        print('RMSE Value is: '+str(self.final_rmse))
        plt.scatter(self.final_df_test,self.final_df_predict)
        plt.title("Scatter Plot of test data and predicted value")
        plt.xlabel(self.final_feature_name)
        plt.ylabel("MEDV")
        plt.show()

    def start_reg(self) :
        for col in self.df_data.columns.values :
            df_x = self.df_data[col].copy().to_frame(col)

            df_x_train, df_x_test = self.split_train_test(df_x)
            self.linear_reg(df_x_train)

            df_y_predict = self.predict(df_x_test)
            rmseVal = self.rmse(df_y_predict)
            r2 = self.r_sq_score(df_y_predict)

            if self.final_r2 == None or (r2>self.final_r2):
                self.final_r2 = r2
                self.final_df_test = df_x_test.values
                self.final_df_predict = df_y_predict.values
                self.final_feature_name = col
                self.final_rmse =  rmseVal
        self.plot()

if __name__ == '__main__':
    lr = Linear_Ref()
    lr.start_reg()

