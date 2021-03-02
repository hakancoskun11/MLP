  # -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

veriler = pd.read_csv('diabetes.csv')

x = veriler.iloc[:,:8].values #bağımsız değişkenler  
y = veriler.iloc[:,8:].values #bağımlı değişken




#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)    




#model oluşturma

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers.core import Activation
from keras.layers.core import Dropout
import keras.optimizers
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test) 

model = Sequential()

model.add(Dense(64,input_dim=8))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(32))
model.add(Activation("softmax"))

earlyStopping = EarlyStopping(monitor="vall_loss",mode="min",verbose=1, patience=25)


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train,y_train,epochs=100,batch_size=10,validation_split=0.13,callbacks=[earlyStopping])

'''
tahmin = np.array([2,122,51,0,0,357.7,0.124,450]).reshape(1,8)
print(model.predict_classes(tahmin))
'''

modelKaybi=pd.DataFrame(model.history.history)
print(modelKaybi.plot()) 




# predict probabilities for test set
yhat_probs = model.predict(X_test, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test, verbose=0)

yhat_probs = yhat_probs
yhat_classes = yhat_classes

cm= confusion_matrix(y_test,yhat_classes)
print(cm)


# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)







