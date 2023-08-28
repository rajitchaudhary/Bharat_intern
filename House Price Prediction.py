# Dataset used for training and testing :- https://www.kaggle.com/datasets/yasserh/housing-prices-dataset

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[51]:


data = pd.read_csv('Housing.csv')


# In[52]:


data.head()


# In[55]:


data.mainroad.replace(('yes', 'no'), (1, 0), inplace=True)
data.guestroom.replace(('yes', 'no'), (1, 0), inplace=True)
data.basement.replace(('yes', 'no'), (1, 0), inplace=True)
data.hotwaterheating.replace(('yes', 'no'), (1, 0), inplace=True)
data.airconditioning.replace(('yes', 'no'), (1, 0), inplace=True)
data.prefarea.replace(('yes', 'no'), (1, 0), inplace=True)
data.head()


# In[56]:


data.furnishingstatus.value_counts()


# In[57]:


pd.get_dummies(data.furnishingstatus)


# In[63]:


data = data.join(pd.get_dummies(data.furnishingstatus)).drop(['furnishingstatus'], axis=1)


# In[64]:


data.info()


# In[65]:


data.columns


# In[66]:


data.describe()


# In[67]:


data.hist()
plt.show()


# In[68]:


sns.pairplot(data)


# In[69]:


X = data[['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad',
       'guestroom', 'basement', 'hotwaterheating', 'airconditioning',
       'parking', 'prefarea', 'furnished', 'semi-furnished', 'unfurnished']]
y = data['price']


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[71]:


model = LinearRegression()


# In[72]:


model.fit(X_train, y_train)


# In[73]:


y_pred = model.predict(X_test)


# In[83]:


sns.displot((y_test-y_pred),bins=50)


# In[84]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[85]:


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[86]:


plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices")
plt.show()


# In[ ]:




