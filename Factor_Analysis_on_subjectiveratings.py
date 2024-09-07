import sklearn
import pandas as pd
from factor_analyzer import FactorAnalyzer
import numpy
from sklearn.preprocessing import StandardScaler


ratings = pd.read_excel('/path/to/subjectiveratings/', names=["/adjectives/used/in/thestudy/"])
print("input shape", numpy.shape(ratings))

#df_features = pd.read_csv('/Users/sirisha/Downloads/WP2Sheet8.csv')
#print("input shape", numpy.shape(df_features), df_features.DataFrame['Kindness','Responsible'])
print("ratings are", ratings)
ratings = StandardScaler().fit_transform(ratings)
print("aftr norm ratings are", ratings)

fa = FactorAnalyzer(n_factors=3,rotation='varimax', method='minres')

fa.fit(ratings)

print(numpy.shape(fa.loadings_))

fa.loadings_

print(fa.loadings_)
numpy.savetxt('/path/to/save/the/resulsts/of/factoranalysis/', fa.loadings_, delimiter=',', fmt='%1.2f')
