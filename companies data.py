#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the data set and setting the independent and dependent variable
comp = pd.read_csv('E:/saved programs/Spyder/Compnies data/data50.csv')
X = comp.iloc[:,1:5].values
y = comp.iloc[:,5].values

#
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Replacing all the empty data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:,3:6])
X[:,3:6] = imputer.transform(X[:,3:6])

#Splitting data into Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2 )

#filling multipul linear regression to the training set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)

#Testing the model
y_pred = regression.predict(X_test)

#Calculating the R Squared Value
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


