import os
import numpy as np
from sklearn.svm import SVC
import cv2
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings(action='ignore')


def read_training_data():
    image_data = []
    target_data = []
    print(os.getcwd())
    for each_letter in os.listdir('Predicted1'):
        for each in os.listdir('Predicted1/'+each_letter):
            path='Predicted1/'+each_letter
            image_path=os.path.join(path,each)
            img_details = cv2.imread(image_path)
            img_details = cv2.cvtColor(img_details,cv2.COLOR_BGR2GRAY)
            ret2,binary_image = cv2.threshold(img_details,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)
    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    accuracy_result = cross_val_score(model, train_data, train_label,cv=num_of_fold)
    print("Cross Validation Result for ", str(num_of_fold), " -fold")
    print(accuracy_result * 100)

def svc_model():
    print("\nSVC Model\n")
    print('reading data')
    image_data, target_data = read_training_data()
    print('reading data completed')
    svc_model = SVC(kernel='linear', probability=True)
    cross_validation(svc_model, 5, image_data, target_data)
    print('training model')
    svc_model.fit(image_data, target_data)
    import pickle
    print("model trained.saving model..")
    filename = './models/svc_model1.sav'
    pickle.dump(svc_model, open(filename, 'wb'))
    print("model saved")

def adaboost():
    print("\nADA Boost Model\n")
    print('reading data')
    image_data, target_data = read_training_data()
    print('reading data completed')
    from sklearn.ensemble import AdaBoostClassifier
    ada_model = AdaBoostClassifier()
    cross_validation(ada_model, 5, image_data, target_data)
    print('training model')
    ada_model.fit(image_data,target_data)
    import pickle
    print("model trained.saving model..")
    filename = './models/ada_model1.sav'
    pickle.dump(ada_model, open(filename, 'wb'))
    print("model saved")

def naivebayes():
    print("\nNaive Bayes model\n")
    print('reading data')
    image_data, target_data = read_training_data()
    print('reading data completed')
    from sklearn.naive_bayes import GaussianNB
    gnb_model = GaussianNB()
    cross_validation(gnb_model, 5, image_data, target_data)
    print('training model')
    gnb_model.fit(image_data,target_data)
    import pickle
    print("model trained.saving model..")
    filename = './models/gnb_model1.sav'
    pickle.dump(gnb_model, open(filename, 'wb'))
    print("model saved")

def logistic():
    print("\nDecision Tree Model\n")
    print('reading data')
    image_data, target_data = read_training_data()
    print('reading data completed')
    from sklearn.tree import DecisionTreeClassifier
    dtc_model = DecisionTreeClassifier()
    cross_validation(dtc_model, 5, image_data, target_data)
    print('training model')
    dtc_model.fit(image_data,target_data)
    import pickle
    print("model trained.saving model..")
    filename = './models/dtc_model1.sav'
    pickle.dump(dtc_model, open(filename, 'wb'))
    print("model saved")

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    print("\nKNN model\n")
    print('reading data')
    image_data, target_data = read_training_data()
    print('reading data completed')
    knn_model = KNeighborsClassifier()
    cross_validation(knn_model, 5, image_data, target_data)
    print('training model')
    knn_model.fit(image_data,target_data)
    import pickle
    print("model trained.saving model..")
    filename = './models/knn_model1.sav'
    pickle.dump(knn_model, open(filename, 'wb'))
    print("model saved")


svc_model()
adaboost()
naivebayes()
logistic()
knn()