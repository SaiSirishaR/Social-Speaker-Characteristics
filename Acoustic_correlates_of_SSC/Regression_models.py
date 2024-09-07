# male competence


import sklearn
from sklearn.svm import SVR
import pandas as pd
import numpy
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression


df_input = pd.read_excel('/path/to/inputfile/', names=["/provide/featnames/])
df_output = pd.read_excel('/path/tosubjectiveratings/', names=["/name/of/the/adjective"]


### Data Normalization

x = StandardScaler().fit_transform(df_input.values)
y = StandardScaler().fit_transform(df_output.values)


#### Linear regression with Leave One out croiss validation

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut


LOO = LeaveOneOut()
los = loo.get_n_splits(x)

mse_error_array = []
rfmse_error_array = []

target_array = []
regressor = LinearRegression()  
lin_reg = regressor.fit(x, y) #training the algorithm


for train_indices, test_indices in LOO.split(x):
     
     target_array.append(y[test_indices][0])
 
preds = cross_val_predict(lin_reg, x, y, cv=los)
mse_error = sklearn.metrics.mean_squared_error(target_array,preds)
mse_error_array.append(mse_error) 

print("total error from Linear Regressor is", mse_error_array)

##### Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(max_depth=None, random_state=0)
rf_model = regr.fit(x,y)
rf_preds = cross_val_predict(rf_model, x, y, cv=los)
rfmse_error = sklearn.metrics.mean_squared_error(target_array,rf_preds)
rfmse_error_array.append(rfmse_error) 

print("total error from RF Regressor is", rfmse_error_array)
    
###### SVM ######

svm_mse_error_array = []
svr_clf = SVR(kernel ='linear',C=1, epsilon=0.2)
svreg = svr_clf.fit(x,y)
sv_preds = cross_val_predict(svreg, x, y, cv=los)
svm_mse_error = sklearn.metrics.mean_squared_error(target_array,sv_preds)
svm_mse_error_array.append(svm_mse_error) 

print("total error with SVR is", np.average(svm_mse_error_array))

    
