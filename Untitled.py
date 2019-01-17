#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
import itertools
from matplotlib.ticker import NullFormatter
import pandas as pd
import matplotlib.ticker as ticker
from sklearn import preprocessing
from scipy import ndimage 
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
get_ipython().run_line_magic('matplotlib', 'inline')


# In[117]:


yelp_db = pd.read_csv('yelp_checkin.csv')
yelp_db1 = pd.read_csv('yelp_business.csv')
yelp_db2 = pd.read_csv('yelp_business_attributes.csv')
yelp_rest = yelp_db.merge(yelp_db1, on ='business_id', how = 'outer')
yelp_all = yelp_rest.merge(yelp_db2, on ='business_id', how = 'outer')
yelp_nydb = yelp_all.loc[yelp_all['state'] == 'NY']
X = yelp_nydb[['checkins','review_count','stars','is_open']].copy()


X = X.values[:,1:]
X = np.nan_to_num(X)
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
y = yelp_nydb['review_count'].values
y[0:5]
X[0:5]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.5, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet

#Train Model and Predict 

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k = 4

#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh



yhat = neigh.predict(X_test)

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[118]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# In[119]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.9)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[120]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# In[ ]:





# In[ ]:




