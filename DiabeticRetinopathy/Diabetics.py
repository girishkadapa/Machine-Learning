#!/usr/bin/env python
# coding: utf-8

# In[84]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[131]:


data = pd.read_csv('messidor_features.arff',error_bad_lines=False)


# In[132]:


col = ['Quality','Pre-screening','MA1','MA2','MA3','MA4','MA5','MA6','MA7',
      'exudates1','exudates2','exudates3','exudates4','exudates5','exudates6','exudates7',
      'macula_opticdisc','opticdisc_diamter','AM/FM','Class_label']


# In[133]:


data.columns = col


# In[134]:


data.info()


# In[ ]:





# In[135]:


data.head()


# In[136]:


data.describe()


# In[126]:





# In[137]:


z = np.abs(stats.zscore(data))


# In[122]:


print(z.shape)


# In[138]:


data = data[(z < 3).all(axis=1)]


# In[139]:


data.describe()


# In[ ]:





# In[71]:


from sklearn.preprocessing import StandardScaler


# In[125]:


prep = StandardScaler()


# In[99]:


X = prep.fit_transform(X)


# In[141]:


X = data.drop(['Class_label','Quality','Pre-screening'],axis=1)


# In[ ]:





# In[142]:


y = data['Class_label']


# In[53]:


from sklearn.model_selection import train_test_split


# In[143]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)


# In[55]:


from sklearn.linear_model import LogisticRegression


# In[150]:


lr = LogisticRegression(solver='lbfgs',random_state=0,max_iter=200)


# In[151]:


lr = LogisticRegression()


# In[152]:


lr.fit(X_train,y_train)


# In[153]:


predict = lr.predict(X_test)


# In[ ]:





# In[154]:


print("Train - ",lr.score(X_train,y_train))
print("Test - ",lr.score(X_test,y_test))


# In[63]:


from sklearn.metrics import classification_report,confusion_matrix


# In[156]:


print(confusion_matrix(y_test,predict))
print(classification_report(y_test,predict))


# In[159]:


from sklearn.neighbors import KNeighborsClassifier


# In[166]:


knn = KNeighborsClassifier()


# In[163]:


knn_error = []
for i in range(2,50):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    knn_predict= knn.predict(X_test)
    knn_error.append(np.mean(y_test!=knn_predict))


# In[183]:


plt.plot(range(2,50),knn_error)


# In[184]:


knn = KNeighborsClassifier(n_neighbors=19)


# In[185]:


knn.fit(X_train,y_train)


# In[186]:


knnpredict=knn.predict(X_test)


# In[187]:


print(confusion_matrix(y_test,knnpredict))
print(classification_report(y_test,knnpredict))


# In[ ]:





# In[167]:


from sklearn.model_selection import GridSearchCV


# In[171]:


neigh={'n_neighbors':np.array(range(2,50))}


# In[188]:


knn_grid=GridSearchCV(KNeighborsClassifier(),neigh,verbose=4,refit=True,cv=3)


# In[189]:


knn_grid.fit(X_train,y_train)


# In[190]:


knn_predict = knn_grid.predict(X_test)


# In[191]:


print(confusion_matrix(y_test,knn_predict))
print(classification_report(y_test,knn_predict))


# In[ ]:




