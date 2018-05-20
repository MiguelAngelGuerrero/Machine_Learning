# Data Preprocessing Template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('../Data/Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

""" # Fix missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3]) """

""" # Label encoding
# Label encode and one-hot encode categorical data of X
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,0] = labelEncoder.fit_transform(X[:,0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()
# Label encode categorical data of y
y = labelEncoder.fit_transform(y) """

# Split dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling (only use if library doesn't take care of this)
from sklearn.preprocessing import StandardScaler
stdScaler_X = StandardScaler()
## Don't scale country information
#X_train[:,3:5] = stdScaler_X.fit_transform(X_train[:,3:5])
#X_test[:,3:5] = stdScaler_X.transform(X_test[:,3:5]) 
# Scale everything
X_train = stdScaler_X.fit_transform(X_train)
X_test = stdScaler_X.transform(X_test) 