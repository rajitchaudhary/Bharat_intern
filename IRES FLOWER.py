#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas
print('pandas: {}'.format(pandas.__version__))


# In[3]:


import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# In[4]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)


# In[5]:


print(dataset.shape)


# In[6]:


print(dataset.head(30))


# In[7]:


print(dataset.describe())


# In[8]:


print(dataset.groupby('class').size())


# In[9]:


dataset.plot(kind='box', subplots = True, layout=(2,2), sharex = False, sharey = False)
plt.show()


# In[10]:


dataset.hist()
plt.show()


# In[11]:


scatter_matrix(dataset)
plt.show()


# In[12]:


array=dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 6
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = validation_size, random_state = seed)


# In[13]:


seed = 6
scoring = 'accuracy'


# In[14]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[15]:


k = 3
knn_classifier = KNeighborsClassifier(n_neighbors=k)


# In[17]:


knn_classifier.fit(X_train_scaled, Y_train)


# In[18]:


Y_pred = knn_classifier.predict(X_test_scaled)


# In[19]:


accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:




