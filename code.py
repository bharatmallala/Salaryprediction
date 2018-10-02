    
import pandas as pd
import numpy as np

#loadig the data sets as a pandas data frame
train_data = pd.read_csv("train_features_2013-03-07.csv")
test_data = pd.read_csv("test_features_2013-03-07.csv")
train_salaries = pd.read_csv("train_salaries_2013-03-07.csv")

#preprocessing the data to remove anamolies and make it ready for buling the model

train_data['salaries'] = train_salaries['salary']

train_data['jobType'] = train_data['jobType'].map({'JANITOR': 0,'JUNIOR':1,'SENIOR':2, 'MANAGER':3, 'VICE_PRESIDENT' :4,'CTO':5,'CFO':6 , 'CEO':7})
train_data['degree'] = train_data['degree'].map({'NONE': 0,'HIGH_SCHOOL':1, 'BACHELORS':2, 'MASTERS' :3, 'DOCTORAL':4})
test_data['jobType'] = test_data['jobType'].map({'JANITOR': 0,'JUNIOR':1,'SENIOR':2, 'MANAGER':3, 'VICE_PRESIDENT' :4,'CTO':5,'CFO':6 , 'CEO':7})
test_data['degree'] = test_data['degree'].map({'NONE': 0,'HIGH_SCHOOL':1, 'BACHELORS':2, 'MASTERS' :3, 'DOCTORAL':4})


X = train_data.iloc[:,:-1].values
Y = train_data.iloc[:,8].values

X_t = test_data.iloc[:,:].values


#checking for missing values in train and test data
train_data.isnull().sum()
test_data.isnull().sum()


#encoding categorical variables using Label and One hot encoder.
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label = LabelEncoder()

X[:,1] = label.fit_transform(X[:,1])
X[:,4] = label.fit_transform(X[:,4])
X[:,5] = label.fit_transform(X[:,5])

X_t[:,1] = label.fit_transform(X_t[:,1])
X_t[:,4] = label.fit_transform(X_t[:,4])
X_t[:,5] = label.fit_transform(X_t[:,5])

X_id = X[:,0]
X1 = X[:,[1,2,3,4,5,6,7]]

X_t_id = X_t[:,0]
X1_t = X_t[:,[1,2,3,4,5,6,7]]


onehot = OneHotEncoder(categorical_features=[0])
X1= onehot.fit_transform(X1).toarray()
X1 = np.delete(X1,0, axis = 1)
X1_t= onehot.fit_transform(X1_t).toarray()
X1_t = np.delete(X1_t,0, axis = 1)


onehot1 = OneHotEncoder(categorical_features=[64])
X1= onehot1.fit_transform(X1).toarray()
X1 = np.delete(X1,64, axis = 1)
X1_t= onehot1.fit_transform(X1_t).toarray()
X1_t = np.delete(X1_t,64, axis = 1)

onehot2 = OneHotEncoder(categorical_features=[72])
X1= onehot2.fit_transform(X1).toarray()
X1 = np.delete(X1,72, axis = 1)
X1_t= onehot2.fit_transform(X1_t).toarray()
X1_t = np.delete(X1_t,72, axis = 1)

#checking correlations among variables to see if any pairs of variables are highly correlated.
X1_df = pd.DataFrame(X1)
temp = X1_df.corr()

#Split the train data into train and test data to check for model performance as we do not have the dependent variable in the Test data set.
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test =  train_test_split(X1,Y, test_size = 0.2 , random_state  = 0)

# Scaling the features to make it much more uniform across the data set and therby improoving the performance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Checking if PCA transformation helps in reducing the existent features.
# It is not quite helpful as there are no few principal components which explains most of the variance in the data.
#from sklearn.decomposition import PCA
#pca = PCA(n_components = None)
#X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_

#Using random forest for feature selection.
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(X1, Y)
rf_pred = rf.predict(X_test)
features = rf.feature_importances_

#Using multiple linear regression on the variables selected.

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train[:,[0,1,3,5,6,11,76,77,78,79]], Y_train)
lr_pred = lr.predict(X_test[:,[0,1,3,5,6,11,76,77,78,79]])
lr_score = r2_score(Y_test, lr_pred)


#Perfoming Support vector Regressor

from sklearn.svm import SVR
sv = SVR(kernel = 'linear')
sv.fit(X_train[0:100000], Y_train[0:100000])
svr_pred = sv.predict(X_test)


#Perfoming Decison tree Regressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=0)
dtr.fit(X_train, Y_train)
dtr_pred = dtr.predict(X_test)


#Using neural network
import keras
from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()
ann.add(Dense(output_dim = 40 , init = 'uniform', activation='relu', input_dim = 80))
ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))
ann.compile(optimizer = 'adam', loss = 'mse')
ann.fit(X_train, Y_train, batch_size= 100 , nb_epoch = 10)
ann_pred = ann.predict(X_test)


#Using bagging regressor on top of linear model.
from sklearn.ensemble import BaggingRegressor as br
brg = br(n_estimators =100, base_estimator = lr)
brg.fit(X_train[:,[0,1,3,5,6,11,76,77,78,79]],Y_train)
brg_pred = brg.predict(X_test[:,[0,1,3,5,6,11,76,77,78,79]])
brg_score = r2_score(Y_test,  brg_pred)

# evaluating model performance using r2_score and Mean square error
from sklearn.metrics import r2_score, mean_squared_error
lr_score = r2_score(Y_test, lr_pred)
lr_error = mean_squared_error(Y_test, lr_pred)
dtr_score = r2_score(Y_test, dtr_pred)
dtr_error = mean_squared_error(Y_test, dtr_pred)
rf_score = r2_score(Y_test, rf_pred)
rf_error = mean_squared_error(Y_test, rf_pred)
svr_score = r2_score(Y_test, svr_pred)
svr_error = mean_squared_error(Y_test, svr_pred)
ann_score = r2_score(Y_test, ann_pred)
ann_error = mean_squared_error(Y_test, ann_pred)


#fitting linear regession for the entire data set
lr1 = LinearRegression()
lr1.fit(X1[:,[0,1,3,5,6,11,76,77,78,79]],Y)
lr1_pred = lr.predict(X1_t[:,[0,1,3,5,6,11,76,77,78,79]])
coefs=lr1.coef_

#summary statistics
import statsmodels.formula.api as sm
l = sm.OLS(Y,X1).fit()
l.summary()


#loading the final predcitions on a Data frame and writing it to a csv file.
test_salaries = pd.DataFrame({'jobId': X_t_id, 'salary' : lr1_pred} )
test_salaries.to_csv('test_salaries.csv', index = False)


