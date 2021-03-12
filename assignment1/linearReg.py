import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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
    
    def initPolyRegDegree(self,deg):
        self.poly_lin = LinearRegression()
        self.poly = PolynomialFeatures(degree=deg)

    def split_train_test(self,data):
        df_train = data.iloc[:self.n_train].copy()
        df_test = data.iloc[self.n_train:].copy()
        return df_train,df_test

    def linear_reg(self,df_train):
        self.reg.fit(df_train,self.df_target_train)

    def poly_reg(self,x):
        df_train_poly_form = self.poly.fit_transform(x)
        self.poly_lin.fit(df_train_poly_form,self.df_target_train.values)

    def predict(self,df_test):
        return pd.DataFrame(self.reg.predict(df_test))

    def rmse(self,df_predict):
        mse = mean_squared_error(self.df_target_test,df_predict)
        return np.sqrt(mse)

    def r_sq_score(self,df_predict):
        return r2_score(self.df_target_test, df_predict,multioutput='variance_weighted')


    def plot(self,final_rmse,final_r2,final_df_predict,final_df_test,final_feature_name,reg_type):
        print('Plotting for '+reg_type)
        print('Most optimum col is: '+final_feature_name)
        print('R-Squared Value is: '+str(final_r2))
        print('RMSE Value is: '+str(final_rmse))
        plt.figure(figsize=(8,6)) #set before plotting
        plt.scatter(final_df_test,self.df_target_test,color='blue')
        plt.plot(final_df_test,final_df_predict,color='red')
        plt.title("Scatter Plot of test data and predicted value - "+reg_type)
        plt.xlabel(final_feature_name)
        plt.ylabel("MEDV")
        plt.show()

    def start_lin_reg(self) :
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
        self.plot(self.final_rmse,self.final_r2,self.final_df_predict,self.final_df_test,self.final_feature_name,'Linear Regression')

    def start_poly_reg(self,reg_type):
        df_x = (self.df_data[self.final_feature_name].copy()).to_frame(self.final_feature_name)
        df_x_train, df_x_test = self.split_train_test(df_x)

        self.poly_reg(df_x_train.values)
        
        x_poly = self.poly.fit_transform(df_x_test.values)
        y_predict = self.poly_lin.predict(x_poly)

        df_y_predict = pd.DataFrame(y_predict)
        rmseVal = self.rmse(df_y_predict)
        r2 = self.r_sq_score(df_y_predict)

        self.plot(rmseVal,r2,y_predict,df_x_test.values,self.final_feature_name,reg_type)
if __name__ == '__main__':
    lr = Linear_Ref()
    lr.start_lin_reg()
    lr.initPolyRegDegree(2)
    lr.start_poly_reg('Polynomial Regression - Degree 2')
    lr.initPolyRegDegree(20)
    lr.start_poly_reg('Polynomial Regression - Degree 20')

