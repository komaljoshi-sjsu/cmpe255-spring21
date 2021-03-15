import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

PROJECT_ROOT_DIR = "."
IMAGE_DIR = "images"

X=None
y=None
sgd_clf=None

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, IMAGE_DIR, fig_id + ".png")
    if os.path.isfile(path):
        os.remove(path)   # Opt.: os.system("rm "+strFile)
    print("\n\nSaving figure...", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def sort_by_target(mnist):
    reorder_index = np.array(sorted([(target, i) for i, target in enumerate(mnist.target)]))[:, 1]
    mnist.data = mnist.data.iloc[reorder_index]
    mnist.target= mnist.target[reorder_index]

def load_and_sort():
    global X
    global y
    mnist = fetch_openml('mnist_784', version=1,cache=True)
    mnist.target = mnist.target.astype(np.int8)
    sort_by_target(mnist)
    X,y = mnist["data"],mnist["target"]
    mnist["data"], mnist["target"]


def random_digit(x,i):
    some_digit = x.iloc[[i]] #taken ith entry of dataframe
    some_digit_image = some_digit.values.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary,
            interpolation="nearest")
    plt.axis("off")

    save_fig("fig_"+str(i))
    plt.show()
    return some_digit

def calculate_cross_val_score(x,y):
    score = cross_val_score(sgd_clf,x,y)
    print(f'cross validation score is: {score}')

def train_predict(some_digit,index):
    global sgd_clf
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X.iloc[:60000], y[:60000]

    X_test, y_test = X.iloc[60000:], y[60000:]

    X_train, y_train = X_train.iloc[shuffle_index],y_train.iloc[shuffle_index]
    # Binary number 5 Classifier
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)

    
    # print prediction result of the given input some_digit
    sgd_clf = SGDClassifier(max_iter=5,tol=-np.infty)
    sgd_clf.fit(X_train, y_train_5)

    prediction = sgd_clf.predict(some_digit)
    print(f'Actual digit value = {y.iloc[index]} \n Is this digit 5?  {prediction}')
    calculate_cross_val_score(X_train,y_train_5)
    
    return X_train,y_train,y_train_5,X_test, y_test_5, y_test

def test() :
    load_and_sort()
    somedigit = random_digit(X,36000)
    X_train,y_train,y_train_5,X_test, y_test_5, y_test = train_predict(somedigit,36000)

    somedigit = random_digit(X_train,15000)
    prediction = sgd_clf.predict(somedigit)
    print(f'Actual digit value = {y_train.iloc[15000]} \n Is this digit 5?  {prediction}')
    calculate_cross_val_score(X_train,y_train_5)

    somedigit = random_digit(X,36500)
    prediction = sgd_clf.predict(somedigit)
    print(f'Actual digit value = {y.iloc[36500]} \n Is this digit 5?  {prediction}')
    calculate_cross_val_score(X_train,y_train_5)

test()