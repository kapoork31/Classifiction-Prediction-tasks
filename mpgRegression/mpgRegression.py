import sklearn as sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics,linear_model
from  matplotlib import pyplot as plt
from knnimpute import knn_impute_few_observed

data = pd.read_csv('autoclean.txt' ,sep = ',') # read in file
data.horsepower[data.horsepower =='?'] = np.NaN  # convert '?' to Na values
dataN = data[['model year','horsepower','origin']] # for knnimputation,decided to use year of make and origin as two variables which can effect horsepower
dataN = dataN.convert_objects(convert_numeric=True)# convert the pandas df into type float 64
dataN = dataN.as_matrix() # convert df into numpy array	

X_imputed = knn_impute_few_observed(dataN, np.isnan(dataN), k= 6) # Knn imputation using knnimpute
X_imputed = np.asarray(X_imputed, dtype=int) 
data['horsepower'] = X_imputed[:,1]

predictors = data[['weight', 'model year','origin','displacement','horsepower']]# usnig these 5 as [predictor variables]
target = data['mpg'] # defnie target data

X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42) # test, train split

clf = linear_model.LinearRegression() # sklearn linear regression.
clf.fit(X_train,y_train) # fit model on training data.
clf.predict(X_test) # use model to predict mpg of test data.
res = clf.score(X_test,y_test) # output the r-squared score of the model.
print (res)
#print (clf.coef_) # coeeficients opf the linar regression. Displays the change in mpg on average based on an increase of 1 unit in the respective predictive variable. 

