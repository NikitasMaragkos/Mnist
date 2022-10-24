import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,f1_score,precision_score,recall_score,accuracy_score
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.neighbors import KNeighborsClassifier

df=pd.DataFrame(pd.read_csv('mnist_train.csv'))
df['new_label']=df['label']%2
df2=pd.DataFrame(pd.read_csv('mnist_test.csv'))
df2['new_label']=df2['label']%2
df=df.drop('label',axis=1)
df2=df2.drop('label',axis=1)
x_train1=df.drop('new_label',axis=1)
y_train=df['new_label']
x_test1=df2.drop('new_label',axis=1)
y_test=df2['new_label']


t0=time()
scaler=MinMaxScaler()
scaler.fit(x_train1)
x_train=scaler.transform(x_train1)
x_test=scaler.transform(x_test1)
print("Standardization done in %0.3fs" % (time() - t0))




t0 = time()
pca=PCA(0.9)
pca.fit(x_train)
x_train=pca.transform(x_train)
x_test=pca.transform(x_test)
print("PCA transformation done in %0.3fs" % (time() - t0))


t0 = time()
model= KNeighborsClassifier(n_neighbors=10,weights='distance',p=2)
model.fit(x_train,y_train)
print("Training with knn done in %0.3fs" % (time() - t0))

predictions_train=model.predict(x_train)
predictions_test=model.predict(x_test)
print("The accuracy in training set is {}".format(accuracy_score(predictions_train,y_train)))#format(model.score(x_train,y_train)))
print("The accuracy in testing set is {}".format(accuracy_score(predictions_test,y_test)))#format(model.score(x_test,y_test)))
print(confusion_matrix(y_test,predictions_test))
print('\n')
print(classification_report(y_test,predictions_test))

