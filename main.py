#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from pandas import read_csv
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, plot_roc_curve
from matplotlib import pyplot
from helper import *
from Model import model
# define the location of the dataset
path = 'BankNote_Authentication.csv'
# load the dataset
df = read_csv(path, header=None)
# summarize shape
print(df.shape)



# In[2]:


print(df.describe())
# plot histograms
df.hist()
pyplot.show()


# In[3]:


# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# determine the number of input features
n_features = X.shape[1]

X_train = X_train.T
y_train = np.expand_dims(y_train, axis = 0)

X_test = X_test.T
y_test = np.expand_dims(y_test,axis=0)
print(X_train.shape,y_train.shape)
print(X_test.shape, y_test.shape)


# In[4]:


layers_dims = [4,10,10, 1] #  3-layer model


# In[14]:


parameters = model(X_train, y_train, layers_dims, num_iterations = 3000, print_cost = True, AF='sigmoid')
print('training accuracy')
pred_train = predict(X_train, y_train, parameters)
print('test accuracy')
pred_test = predict(X_train, y_train, parameters)


# In[15]:


parameters = model(X_train, y_train, layers_dims, num_iterations = 3000, print_cost = True, AF='Lrelu')
print('training accuracy')
pred_train = predict(X_train, y_train, parameters)
print('test accuracy')
pred_test = predict(X_train, y_train, parameters)


# In[16]:


parameters = model(X_train, y_train, layers_dims, num_iterations = 3000, print_cost = True, AF='relu')
print('training accuracy')
pred_train = predict(X_train, y_train, parameters)
print('test accuracy')
pred_test = predict(X_train, y_train, parameters)


# In[17]:


parameters = model(X_train, y_train, layers_dims, num_iterations = 3000, print_cost = True, AF='tanh')
print('training accuracy')
pred_train = predict(X_train, y_train, parameters)
print('test accuracy')
pred_test = predict(X_train, y_train, parameters)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:





# In[ ]:




