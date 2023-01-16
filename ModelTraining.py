#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install sklearn ')
get_ipython().system('pip install pickle')


# In[1]:
pip install pandas

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


# In[2]:


df=pd.read_csv('diabetes.csv')


# In[3]:


X=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[4]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


# In[5]:


classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)


# In[6]:


y_pred=classifier.predict(X_test)


# In[7]:


from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)


# In[8]:


score


# In[9]:


pickle_out = open("classifier.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()


# In[11]:


classifier.predict([[2,3,4,1]])


# In[ ]:




