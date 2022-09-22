#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing the essential liabraries


# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


# In[2]:


#Upload the dataseta for Housing price ,GDP,Unemployment,Nominal Income,House for sale, Mortage rate,Federal Fund and land permit


# In[3]:


ds= pd.read_csv("housing price data.csv")
ds


# In[4]:


ds.describe()


# In[5]:


print(ds.isnull().sum())


# In[6]:


#Convert the into data Frame


# In[7]:


df=pd.DataFrame(ds)


# In[8]:


df


# In[9]:


df.shape #shape of data frame


# In[10]:


df.size # size of data frame


# In[11]:


#Convert the int data into float


# In[28]:


df['Nominal Income']=df['Nominal Income'].astype(float)
df['House for sale']=df['House for sale'].astype(float)
df['Land Permit']=df['Land Permit'].astype(float)


# In[29]:


df


# In[ ]:


#Creating a pair plot


# In[30]:


sns.pairplot(df) 


# In[ ]:


#Creating a heat map


# In[31]:


sns.heatmap(df.corr(),annot=True,cmap='PiYG')


# In[ ]:


#take the undependent variable for X


# In[32]:


X=df.iloc[:,1:8]


# In[33]:


X


# In[ ]:


#Housing price index is dependent variable


# In[34]:


y=df.loc[:,"Housing_Price_Index"]


# In[35]:


y


# In[ ]:


#Splitting the test and train data


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15,random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:


#train the data


# In[37]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[38]:


X_train


# In[ ]:


#Linear Regression Model


# In[23]:


reg = LinearRegression()
reg.fit(X_train, y_train)
y_predict = reg.predict(X_test)


# In[39]:


y_predict


# In[ ]:


#Comparing the value and finding out the accuracy


# In[45]:


print(reg.coef_)
print(reg.intercept_)
print(reg.score(X_train, y_train))
print(reg.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
print(y_test.describe())
print("Accuracy",reg.score(X_test,y_test)*100)


# In[ ]:


#Random forest regressor


# In[42]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dt=RandomForestRegressor(n_estimators= 10,criterion="mae") 
dt.fit(X_train,y_train)
y_predicted = dt.predict(X_test)
accuracy = (dt.score(X_test,y_test))*100
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predicted))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predicted))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
print(y_test.describe())


# In[ ]:


#Accuracy rate gained after Random regressor model


# In[43]:


accuracy


# In[ ]:




