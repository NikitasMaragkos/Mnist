import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import GridSearchCV
from time import time
import pickle



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
x_train1=scaler.transform(x_train1)
x_test1=scaler.transform(x_test1)
print("Standardization done in %0.3fs" % (time() - t0))




t0 = time()
pca=PCA(0.9)
pca.fit(x_train1)
x_train=pca.transform(x_train1)
x_test=pca.transform(x_test1)
print("PCA transformation done in %0.3fs" % (time() - t0))
print(x_train.shape)
 

#param_grid = { "C" : [10,11,12,9] , "gamma" : [0.12,0.13,0.14]}
#rf = SVC(kernel='rbf')
#gs = GridSearchCV(estimator=rf,n_jobs=-1, param_grid=param_grid, scoring='accuracy',cv=2,verbose=1)
#gs = gs.fit(x_train, y_train)

#model=SVC(C=gs.best_params_["C"], kernel=rbf,gamma=gs.best_params_["gamma"])

t0 = time()
model=SVC(C=10, kernel='rbf',degree=1, gamma=0.01567)
model.fit(x_train,y_train)
print("Training with SVC done in %0.3fs" % (time() - t0))
print(len(model.support_vectors_))




predictions_train=model.predict(x_train)
predictions_test=model.predict(x_test)
print(predictions_test[0],y_test[0])


print("The accuracy in training set is {}".format(accuracy_score(predictions_train,y_train))),0
print("The accuracy in testing set is {}".format(accuracy_score(predictions_test,y_test)))
print(confusion_matrix(y_test,predictions_test))
print('\n')
print(classification_report(y_test,predictions_test))


j=0
for i in range(0,10000):
    g=predictions_test[i] - y_test[i]
    if (g !=0):
        first_image = x_test1[i][:]
        first_label = predictions_test[i]
        plottable_image = np.reshape(first_image,(28, 28))
        plt.imshow(plottable_image, cmap='gray_r')
        plt.title('Digit Label {} and Digit Predicted: {}'.format(y_test[i],first_label))
        plt.show()
        
    else:
        if(j<10):
            first_image = x_test1[i][:]
            first_label = predictions_test[i]
            plottable_image = np.reshape(first_image,(28, 28))
            plt.imshow(plottable_image, cmap='gray_r')
            plt.title('Digit Label {} and Digit Predicted: {}'.format(y_test[i],first_label))
            plt.show()
            j=j+1
       
        




