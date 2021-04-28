from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn import model_selection

np.random.seed(10)

def load_data():
    # fetch photographs from the LFW dataset
    faces = fetch_lfw_people(min_faces_per_person=60)
    print('data loaded')
    print(faces.target_names)
    print(faces.images_shape)

    #We will ignore relative pixel position and treat each photograph as a  62×47=2914  dimensional datapoint
    photos = lfw_people.data
    print(photos.shape)

    #label of each photograph
    labels = lfw_people.target

    photos_train, photos_test, labels_train, labels_test = model_selection.train_test_split(photos, labels, test_size=0.25,random_state=42)
    
    #reduce the number of dimensions while preserving the variance
    pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
    photos_train_pca = pca.transform(photos_train)
    photos_test_pca = pca.transform(photos_test)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)


    C_range = np.logspace(-1, 5, 4)
    gamma_range = np.logspace(-3, 0, 4)
    param_grid = dict(gamma=gamma_range, C=C_range)
    gsv = GridSearchCV(estimator=model, param_grid)

    gsv = gsv.fit(photos_train_pca, labels_train)

    # predict labels of our test set
    labels_pred = gsv.predict(photos_test_pca)