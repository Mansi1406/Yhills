#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


traindf = pd.read_csv('TaxiFare.csv', na_values={"pickup_datetime":"43"})
traindf.head()
traindf = traindf.drop(["unique_id"], axis=1)
traindf.head()


# In[3]:


traindf.isna().sum()


# In[4]:


traindf.describe()


# In[5]:


traindf['amount'] = pd.to_numeric(traindf['amount'], errors='coerce')


# In[6]:


traindf['date_time_of_pickup'] = pd.to_datetime(traindf['date_time_of_pickup'], format='%Y-%m-%d %H:%M:%S UTC')


# In[7]:


traindf['Hour'] = traindf['date_time_of_pickup'].dt.hour
traindf['minute'] = traindf['date_time_of_pickup'].dt.minute
traindf['date'] = traindf['date_time_of_pickup'].dt.day
traindf['day'] = traindf['date_time_of_pickup'].dt.dayofweek
traindf['month'] = traindf['date_time_of_pickup'].dt.month
traindf['year'] = traindf['date_time_of_pickup'].dt.year


# In[8]:


traindf.head()


# In[9]:


traindf.drop(traindf[traindf['date_time_of_pickup'].isna()].index, inplace=True)


# In[10]:


traindf.drop('date_time_of_pickup', axis=1, inplace=True)


# In[11]:


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')


imputer.fit(traindf[['amount','no_of_passenger']])
traindf[['amount','no_of_passenger']] = imputer.transform(traindf[['amount','no_of_passenger']])


# In[12]:


traindf.isna().sum()


# In[13]:


traindf.shape


# In[14]:


plt.figure(figsize=(10,5))
sns.boxplot(y=traindf['no_of_passenger'])


# In[15]:


plt.figure(figsize=(10,5))
sns.boxplot(y=traindf['amount'])


# In[16]:


Q1 = traindf['no_of_passenger'].quantile(0.25)
Q3 = traindf['no_of_passenger'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
print('Lower :',lower)
print('Upper :',upper)


# In[17]:


traindf['no_of_passenger'].sort_values(ascending=True).head(10)


# In[18]:


traindf['amount'].sort_values(ascending=True).head(10)


# In[19]:


traindf.drop(traindf[traindf['amount'] > 150 ].index, inplace = True)
traindf.drop(traindf[traindf['amount'] == 0 ].index, inplace = True)
traindf.drop(traindf[traindf['amount'].isna() ].index, inplace = True)


# In[20]:


traindf.drop(traindf[traindf['latitude_of_pickup'] > 90].index, inplace=True)
traindf.drop(traindf[traindf['latitude_of_pickup'] < -90].index, inplace=True)
traindf.drop(traindf[traindf['latitude_of_dropoff'] > 90].index, inplace=True)
traindf.drop(traindf[traindf['latitude_of_dropoff'] < -90].index, inplace=True)


# In[21]:


traindf.drop(traindf[traindf['longitude_of_pickup'] > 180].index, inplace=True)
traindf.drop(traindf[traindf['longitude_of_pickup'] < -180].index, inplace=True)
traindf.drop(traindf[traindf['longitude_of_dropoff'] > 180].index, inplace=True)
traindf.drop(traindf[traindf['longitude_of_dropoff'] < -180].index, inplace=True)


# In[22]:


traindf.shape


# In[23]:


traindf.isna().sum()


# In[24]:


from math import *

def haversine(a):
    lon1=a[0]
    lat1=a[1]
    lon2=a[2]
    lat2=a[3]
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c =  2 * asin(sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371* c
    return km


# In[25]:


traindf['distance'] = traindf[['longitude_of_pickup','latitude_of_pickup','longitude_of_dropoff','longitude_of_dropoff']].apply(haversine,axis=1)


# In[26]:


traindf.drop(['longitude_of_pickup','latitude_of_pickup','longitude_of_dropoff','latitude_of_dropoff'], axis=1, inplace=True)


# In[27]:


traindf['Hour'] = traindf['Hour'].astype('int64')
traindf['minute'] = traindf['minute'].astype('int64')
traindf['date'] = traindf['date'].astype('int64')
traindf['day'] = traindf['day'].astype('int64')
traindf['month'] = traindf['month'].astype('int64')
traindf['year'] = traindf['year'].astype('int64')


# In[28]:


traindf.head()


# In[29]:


traindf['distance'].sort_values(ascending=True).head(10)


# In[30]:


traindf.drop(traindf[traindf['distance'] >= 130].index, inplace=True)
traindf.drop(traindf[traindf['distance'] <= 0 ].index, inplace=True)


# In[31]:


traindf.head()


# In[32]:


plt.figure(figsize=(10,5))
sns.boxplot(y=traindf['no_of_passenger'])


# In[33]:


plt.figure(figsize=(10,5))
sns.boxplot(y=traindf['amount'])


# In[34]:


sns.boxplot(y=traindf['distance'])


# In[35]:


traindf.describe()


# In[36]:


traindf.columns


# In[37]:


traindf['no_of_passenger'] = traindf['no_of_passenger'].astype('int64')


# In[38]:


traindf.head()


# In[39]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(traindf.drop('amount', axis=1),
                                                   traindf['amount'], test_size=0.4, random_state = 42)


# In[40]:


X_test


# In[41]:


X_train


# In[42]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(criterion='friedman_mse', random_state=42)


# In[43]:


model.fit(X_train, y_train)


# In[44]:


y_pred_test = model.predict(X_test)
y_pred_test


# In[45]:


from sklearn.metrics import mean_absolute_error,mean_squared_error



print("MAE = ",mean_absolute_error(y_test, y_pred_test))
print("MSE = ",mean_squared_error(y_test, y_pred_test))



# In[ ]:





# In[ ]:




