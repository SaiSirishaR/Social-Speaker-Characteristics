# Classification models, SVM, NN

# SVM, NNs for female voices
from numpy import loadtxt
import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

# load the dataset
dataset = pd.read_excel('/path/to/datafile/')
labels = loadtxt('/path/to/labels')
#print("dataset is", dataset,"shape is", numpy.shape(dataset[:,5]), "labels are", labels)

X = StandardScaler().fit_transform(dataset.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True)

################# SVC ######################

clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print("svm accuracy", accuracy_score(y_test, y_pred))

############################################

################# NN ######################

def create_model():
  
 # create model
 model = Sequential()
 model.add(Dense(12, input_dim=11, activation='relu'))
 model.add(Dense(8, activation='relu'))
 model.add(Dense(1, activation='sigmoid'))
  
 # Compile model
 model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
 return model


model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=16, verbose=0)

kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold)
print(results.mean())

############################################
