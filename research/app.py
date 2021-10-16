import json
import numpy as np
import pandas as pd
import joblib

import cv2
from skimage.feature import hog
from sklearn.svm import LinearSVC
from keras.datasets import mnist
from sklearn.metrics import accuracy_score

#load data
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#cho x_train
X_train_feature = []
for i in range(len(X_train)):
    feature = hog(X_train[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_train_feature.append(feature)
X_train_feature = np.array(X_train_feature,dtype = np.float32)

#cho x_test
X_test_feature = []
for i in range(len(X_test)):
    feature = hog(X_test[i],orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),block_norm="L2")
    X_test_feature.append(feature)
X_test_feature = np.array(X_test_feature,dtype=np.float32)

model = LinearSVC(C=10)
model.fit(X_train_feature,y_train)

#lưu weight 
joblib.dump(X_train_feature, "./X_train_feature.joblib", compress=True)
joblib.dump(X_test_feature, "./X_test_feature.joblib", compress=True)
joblib.dump(model, "./model.joblib", compress=True)


# y_pre = model.predict(X_test_feature)
# print(accuracy_score(y_test,y_pre))
