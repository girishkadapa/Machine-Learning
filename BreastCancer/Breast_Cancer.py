#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[2]:


def loaddata():
    col = ['ID Number','Diagnosis','radius','texture','perimeter','area','smoothness','compactness','concavity','concave points','symmetry','fractal dimension'
        ,'radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave points2','symmetry2','fractal dimension2'
        ,'radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave points3','symmetry3','fractal dimension3']

    data = pd.read_csv('wdbc.data',error_bad_lines=False)
    data.columns = col
    return data


# In[3]:


def printaccuracy(y_test,predict,model):
    print(model," report")
    print("-------------------------------------")
    print(" ")
    print(confusion_matrix(y_test,predict))
    print(classification_report(y_test,predict))
    print(" ")
    print("-------------------------------------")
    print(" ")


# In[4]:


def normalizedata(X):
    SS = StandardScaler()
    X = SS.fit_transform(X)
    print("Normalization done")
    


# In[5]:


def PC(components,x):
    cols = []
    pca = PCA(n_components=components)
    pc = pca.fit_transform(x)
    for i in range(components):
        cols.append('pc'+str(i))
    pc_data = pd.DataFrame(data = pc, columns = cols)
    return pc_data


# In[ ]:





# In[6]:


def removeoutliers(data,inplace=False):
    prev_rows = len(data)
    data_copy = data.copy()
    z_score = np.abs(stats.zscore(data_copy))
    data_copy = data_copy[(z_score < 3).all(axis=1)]
    if inplace:
        data=data_copy
    #print("Before removing outliers , rows - ", prev_rows)
    #print("After removing outliers , rows -", len(data_copy))
    #print("Number of records deleted - ", (prev_rows - len(data_copy))#)


# In[7]:


def validatecols(data):
    if len(data.columns) == len(col):
        return True
    else:
        return False


# In[8]:


def validatedatatypes(trained, newdata):
    for i in range(trained.columns):
        if trained[trained.columns[i]] != newdata[newdata.columns[i]]:
            return False
    return True    


# In[9]:


def preprocess():
    data.drop('ID Number',axis=1,inplace=True)
    data['Diagnosis'] = pd.Categorical(data['Diagnosis']).codes
    removeoutliers(data,inplace=True)
    X = data.drop('Diagnosis',axis=1)
    X_copy = X.copy()
    y= data['Diagnosis']
    normalizedata(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
    return X_train,X_test,y_train,y_test,X_copy,y


# In[10]:


def k_vs_error_graph():
    knn_error = []
    for i in range(2,50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train,y_train)
        knn_predict= knn.predict(X_test)
        knn_error.append(np.mean(y_test!=knn_predict))
    plt.plot(range(2,50),knn_error)
    plt.xlabel("K value")
    plt.ylabel("Error")


# In[11]:


def logisticregression():
    lr = LogisticRegression(solver='lbfgs',max_iter=10000,random_state=0)
    lr.fit(X_train,y_train)
    lr_predict = lr.predict(X_test)
    printaccuracy(y_test,lr_predict,"Logistic Regression")
    return f1_score(y_test,lr_predict)


# In[12]:


def KNN():
    neighbors={'n_neighbors':np.array(range(2,50))}
    knn_grid=GridSearchCV(KNeighborsClassifier(),neighbors,verbose=False,refit=True,cv=3)
    knn_grid.fit(X_train,y_train)
    #knn_grid.best_params_
    knn_predict = knn_grid.predict(X_test)
    printaccuracy(y_test,knn_predict,"KNN")
    return f1_score(y_test,knn_predict)


# In[13]:


def SVM():
    svm = SVC(kernel='rbf',random_state=0)
    params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
    svm_grid = GridSearchCV(svm, params,cv=3,verbose=False,return_train_score=True)
    svm_X = X.copy()
    svm_X = PC(2,svm_X)
    svm_X_train,svm_X_test,svm_y_train,svm_y_test=train_test_split(svm_X,y,test_size=0.3)
    svm_grid.fit(svm_X_train,svm_y_train)
    svm_predict = svm_grid.predict(svm_X_test)
    printaccuracy(svm_y_test,svm_predict,"SVM")
    return f1_score(svm_y_test,svm_predictict)


# In[14]:


def DecisionTree():
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train,y_train)
    dt_predict = dt.predict(X_test)
    printaccuracy(y_test,dt_predict,"Decision Tree")
    return f1_score(y_test,dt_predict)


# In[15]:


def RandomForest():
    rf = RandomForestClassifier(random_state=0)
    params = { 'n_estimators' : np.arange(10,100,10), 'max_depth' : np.arange(5,50,5)}
    rf_grid = GridSearchCV(rf, params, verbose=False, cv=3)
    rf_grid.fit(X_train,y_train)
    rf_predict = rf_grid.predict(X_test)
    printaccuracy(y_test,rf_predict,"Random Forest")
    rf
    return f1_score(y_test,rf_predict)


# In[16]:


def Adaboost():
    ab = AdaBoostClassifier(random_state=0)
    params = { 'n_estimators' : np.arange(10,100,10)}
    ab_grid = GridSearchCV(ab, params, verbose=False, cv=3)
    ab_grid.fit(X_train,y_train)
    ab_predict = ab_grid.predict(X_test)
    printaccuracy(y_test,ab_predict,"Adaboost")
    return f1_score(y_test,ab_predict)


# In[17]:


def GaussionNB():
    gnb = GaussianNB()
    gnb.fit(X_train,y_train)
    gnb_predict = gnb.predict(X_test)
    printaccuracy(y_test,gnb_predict,"GaussionNB")
    return f1_score(y_test,gnb_predict)


# In[18]:


def NueralNetwork():
    nn = MLPClassifier(solver='sgd',random_state=0)
    params = {
    'hidden_layer_sizes': np.arange(50,150,20),
    'learning_rate': ['constant','adaptive'],
    'max_iter': np.arange(200,300,50)
    }
    #'hidden_layer_sizes': [(100,50), (50,20), (20,10)],
    #'hidden_layer_sizes': np.arange(10,100,20)
    # 'activation': ['tanh', 'relu'],
    # 'alpha': 10.0 ** -np.arange(1, 5),
    #'solver': ['sgd', 'adam'],
    nn_grid = GridSearchCV(nn, params, cv=3,verbose=False)
    nn_grid.fit(X_train,y_train)
    nn_predict = nn_grid.predict(X_test)
    printaccuracy(y_test,nn_predict,"Nueral Network")
    return f1_score(y_test,nn_predict)


# In[ ]:





# In[ ]:





# In[ ]:





# In[19]:


def EvaluateModels(f1scores):
    models=['Logistic Regression',
            'K Nearest Neighbours',
            'Support Vector Machine',
            'Decision Tree',
            'Random Forest',
            'AdaBoost',
            'Guassian Naive Bayes',
            'Nueral Network'
           ]

    print("        Model Results    ")
    print(" --------------------------- ")
    for i in range(len(f1scores)):
        print(models[i]," : f1 score - ",f1scores[i])
    print(" -----------------------------")
    print("Best model")
    idx = f1scores.index(np.max(f1scores))
    print(models[idx] ," : f1 score - ", np.max(f1scores))
    
    
    
    
    


# In[20]:


def trainmodels():
    f1scores=[]
    f1scores.append(logisticregression())
    f1scores.append(KNN())
    f1scores.append(SVM())
    f1scores.append(DecisionTree())
    f1scores.append(RandomForest())
    f1scores.append(Adaboost())
    f1scores.append(GaussionNB())
    f1scores.append(NueralNetwork())
    return f1scores


# In[ ]:





# In[21]:


data = loaddata()
X_train,X_test,y_train,y_test,X,y = preprocess()
f1scores = trainmodels()
EvaluateModels(f1scores)

