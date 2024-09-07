### Backward elimination algorithm for prediction of acoustic correlates

import numpy
import statsmodels.regression.linear_model as sm 
# add a column of ones as integer data type 
df_input = pd.read_excel('/path/tofeatfolder/')
print(df_input.head())
x = df_input[["/add/featnames/"]]
df_output = pd.read_excel('/pathto/subjectiveratings')
y = df_output[["Warmth"]]
print(numpy.shape(df_input))

x = np.append(arr = np.ones((18,1)).astype(int),  
              values = df_input, axis = 1) 
# choose a Significance level usually 0.05, if p>0.05 
#  for the highest values parameter, remove that value 
x_opt = x[:, [0, 1, 2, 3, 4, 5]] 
ols = sm.OLS(endog = y, exog = x_opt).fit() 
ols.summary() 
