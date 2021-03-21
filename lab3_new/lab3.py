import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0, names=col_names, usecols=col_names)
        print(self.pima.head())
        self.X_test = None
        self.y_test = None

    def define_feature(self):
        feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
        X = self.pima[feature_cols]
        y = self.pima.label
        return X, y

    def define_feature_selection(self):
        cor = self.pima.corr().abs()['label'].sort_values(ascending=False)

        new_cor = cor.drop(labels = ['label']).head(3)
        df_x = pd.DataFrame()

        for key,value in new_cor.items():
            df_x[key] = self.pima[key]

        return df_x,self.pima.label

    def feature_engineering(self):
        #TODO:
        pass
    
    def train(self,feature_processing,logreg):
        # split X and y into training and testing sets
        X, y = feature_processing()
        X_train, self.X_test, y_train, self.y_test = train_test_split(X, y, random_state=0)
        # train a logistic regression model on the training set
        logreg.fit(X_train, y_train)
        return logreg
    
    def predict(self,feature_processing,logreg):
        model = self.train(feature_processing,logreg)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class


    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)


    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()
    
    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    classifer = DiabetesClassifier()

    #baseline solution
    print('\n\n******************** Baseline Solution ********************')
    logreg =  LogisticRegression()
    result = classifer.predict(classifer.define_feature,logreg)
    print(f"\nPredicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"\nscore={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"\nconfusion_matrix ->\n{con_matrix}")

    #after feature selection and applying different solver to best features : Solution 1
    print('\n\n******************** Solution 1 - RFE ********************')
    logreg = RFE(LogisticRegression(solver='lbfgs'),n_features_to_select=3)
    result = classifer.predict(classifer.define_feature,logreg)
    print(f"\nPredicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"\nscore={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"\nconfusion_matrix ->\n{con_matrix}")

    #after feature selection : Solution 2
    print('\n\n******************** Solution 2 - Feature Selection using Correlation ********************')
    logreg = LogisticRegression()
    result = classifer.predict(classifer.define_feature_selection,logreg)
    print(f"\nPredicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"\nscore={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"\nconfusion_matrix ->\n{con_matrix}")


    #after feature selection and applying different solver to best features : Solution 3
    print('\n\n******************** Solution 3 - liblinear solver in best features ********************')
    logreg = LogisticRegression(solver='liblinear')
    result = classifer.predict(classifer.define_feature_selection,logreg)
    print(f"\nPredicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"\nscore={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"\nconfusion_matrix ->\n{con_matrix}")