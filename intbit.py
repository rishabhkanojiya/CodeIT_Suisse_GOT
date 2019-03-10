#!/usr/bin/env python
# coding: utf-8

# In[104]:


#This project was done in Jupyter Notebook


# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from  keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import time


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('train.csv')


# In[4]:


train.head()


# In[5]:


X = train


# In[69]:


X.info()


# In[70]:


#sns.heatmap(X.isnull())


# In[8]:


#sns.pairplot(X_train)


# In[9]:


X.info()


# In[71]:


X.columns


# In[11]:


cols = ['soldierId','shipId','attackId','horseRideDistance']


# In[12]:


X=X.drop(cols,axis=1)


# In[13]:


y = X['bestSoldierPerc']


# In[14]:


X =X.drop(['bestSoldierPerc'],axis=1)


# In[15]:


from sklearn.preprocessing import StandardScaler


# In[16]:


st = StandardScaler()


# In[79]:


X = st.fit_transform(X)


# In[80]:


X


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.15,random_state = 43)


# In[21]:


n_col = X_train.shape[1]
n_col


# In[22]:


from keras.layers import BatchNormalization
import keras.backend as K


# In[23]:


K.clear_session()


# In[24]:


model = Sequential()
model.add(Dense(12,input_shape = (n_col,),activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(12,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))
model.compile(Adam(lr=0.01),loss='mean_absolute_error',metrics=['accuracy'],)


# In[25]:


from keras.callbacks import EarlyStopping


# In[26]:


early_stopping_monitor = EarlyStopping(patience=3)


# In[46]:


model.fit(X_train,y_train, validation_split=0.2, epochs=25)


# In[62]:


#model.save('intbit.h5')


# In[47]:


pred = model.predict(X_test)


# In[48]:


pred


# In[39]:


#from sklearn.ensemble import RandomForestRegressor


# In[40]:


#rfr = RandomForestRegressor(n_estimators=100)


# In[41]:


#rfr.fit(X_train,y_train)


# In[42]:


#pred = rfr.predict(X_test)


# In[43]:


from sklearn.metrics import mean_absolute_error


# In[88]:


mean_absolute_error(y_test,pred)


# In[61]:


pred


# In[52]:


y_test


# In[63]:


test = pd.read_csv('test.csv')


# In[91]:


soid = test['soldierId']


# In[67]:


test.info()


# In[68]:


X = test


# In[73]:


X.columns


# In[74]:


cols = ['Unnamed: 0', 'index','soldierId','shipId','attackId','horseRideDistance']


# In[75]:


X=X.drop(cols,axis=1)


# In[ ]:


X = st.fit_transform(X)


# In[ ]:


X


# In[81]:


predt = model.predict(X)


# In[82]:


predt


# In[94]:


pd.DataFrame(data=predt,index=soid,columns=['bestSoldierPerc'])


# In[95]:


subsp = pd.read_csv('sample_submission.csv')


# In[99]:


subsp['bestSoldierPerc'] = predt


# In[100]:


subsp


# In[103]:


subsp.to_csv('final.csv',index=False)

