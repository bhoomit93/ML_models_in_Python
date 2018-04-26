# -*- coding: utf-8 -*-
"""
@author: patel

Data Pre-processing
"""

#importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import the dataset from csv

dataset = pd.read_csv("Data.csv")
"""

dataset
Out[10]: 
   Country   Age   Salary Purchased
0   France  44.0  72000.0        No
1    Spain  27.0  48000.0       Yes
2  Germany  30.0  54000.0        No
3    Spain  38.0  61000.0        No
4  Germany  40.0      NaN       Yes
5   France  35.0  58000.0       Yes
6    Spain   NaN  52000.0        No
7   France  48.0  79000.0       Yes
8  Germany  50.0  83000.0        No
9   France  37.0  67000.0       Yes

"""
X= dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values


#Taking care of missing Data
from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer= imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


#Taking Care of Categorical Variable

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder_X=LabelEncoder()
X[:,0] = encoder_X.fit_transform(X[:,0])
hotencoder = OneHotEncoder(categorical_features=[0])
X = hotencoder.fit_transform(X).toarray()

encoder_y=LabelEncoder()
y= encoder_y.fit_transform(y)

# Split into train and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

#Feature Scaling
#Standardisation(x-mean/standard dev.) & Normalization(x-Min/Max-Min)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform (X_test)
