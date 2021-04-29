from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn import model_selection
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(10)

def plot_gallery(images, titles, names_actual, h, w, n_row=3, n_col=4, fig_title=None):
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        fc = 'black'
        if titles[i]!=names_actual[i] :
            fc = 'red'
        title = "Predicted: "+titles[i]+"\nActual: "+names_actual[i]
        ax.set_title(titles[i], size=12,color=fc)
        plt.xticks(())
        plt.yticks(())
    if fig_title: 
        fig.suptitle(fig_title+'\n', fontsize=20)

    plt.show(block=True)

def heatmap(cm):
    sns.heatmap(cm, annot=True)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show(block=True)

def load_data():
    # fetch photographs from the LFW dataset
    faces = fetch_lfw_people(min_faces_per_person=60)
    names = faces.target_names
    # get number of pictures and dimensions
    n_samples, h, w = faces.images.shape
    print('data loaded')
    print(names)

    #We will ignore relative pixel position and treat each photograph as a  62×47=2914  dimensional datapoint
    photos = faces.data

    #label of each photograph
    labels = faces.target

    photos_train, photos_test, labels_train, labels_test = model_selection.train_test_split(photos, labels, test_size=0.25,random_state=42)
    
    #reduce the number of dimensions while preserving the variance
    pca = RandomizedPCA(n_components=150, svd_solver='randomized', whiten=True, random_state=42).fit(photos_train)
    photos_train_pca = pca.transform(photos_train)
    photos_test_pca = pca.transform(photos_test)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)

    C_range = np.logspace(-1, 5, 4)
    gamma_range = np.logspace(-3, 0, 4)
    param_grid = dict(svc__gamma=gamma_range, svc__C=C_range)
    gsv = model_selection.GridSearchCV(model, param_grid)

    gsv = gsv.fit(photos_train_pca, labels_train)

    # predict labels of our test set
    labels_pred = gsv.predict(photos_test_pca)
    # get indices to randomly look at our predictions
    r_ind = np.random.choice(photos_test.shape[0], size=24, replace=False)

    # get pictures and respective predictions to look at
    s_photos = photos_test[r_ind]
    labels_actual = labels_test[r_ind]
    labels_pred_sample = labels_pred[r_ind]
    names_pred = names[labels_pred_sample]
    names_actual = names[labels_actual]


    print(metrics.classification_report(labels_test, labels_pred, target_names=names))

    plot_gallery(s_photos, names_pred, names_actual, h, w, n_row=4, n_col=6, fig_title="Predictions")

    cm = metrics.confusion_matrix(names_pred,names_actual,labels=names)
    heatmap(cm)

if __name__ == "__main__":
    load_data()