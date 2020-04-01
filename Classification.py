#!/usr/bin/env python
# coding: utf-8

# In[16]:


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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc


class entryPoint():

    def printaccuracy(self,y_test,predict,model):
        print(model," report")
        print("-------------------------------------")
        print(" ")
        print(" Confusion Matrix " ,confusion_matrix(y_test,predict))
        print(classification_report(y_test,predict))
        print(" ")
        print("-------------------------------------")
        print(" ")
    
    def normalizedata(self,X):
        SS = StandardScaler()
        X = SS.fit_transform(X)
        print("Normalization done")
        return X

    def removeoutliers(self,data,inplace=False):
        prev_rows = len(data)
        data_copy = data.copy()
        z_score = np.abs(stats.zscore(data_copy))
        data_copy = data_copy[(z_score < 3).all(axis=1)]
        if inplace:
            data=data_copy
        print("Before removing outliers , rows - ", prev_rows)
        print("After removing outliers , rows -", len(data_copy))
        print("Number of records deleted - ", (prev_rows - len(data_copy)))
        return data_copy
    
    def PC(self,components,x):
        cols = []
        pca = PCA(n_components=components)
        pc = pca.fit_transform(x)
        for i in range(components):
            cols.append('pc'+str(i))
        pc_data = pd.DataFrame(data = pc, columns = cols)
        return pc_data


    def train_split(self,X,y,test_size=0.2,random_state=0):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
        return X_train,X_test,y_train,y_test

    def knn(self,X_train,y_train,X_test,y_test):
        print("Knn")
        knn_error = []
        for i in range(2,10):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train,y_train)
            knn_predict= knn.predict(X_test)
        print(type(knn_predict))
        print(type(y_test))
        knn_error.append(np.mean(y_test!=knn_predict))
        plt.plot(range(2,50),knn_error)
        plt.xlabel("K value")
        plt.ylabel("Error")
    
    def knn_grid_search(self,X_train,y_train,X_test,y_test,inp_params):
        print("Knn Grid Search Starting...")
        knn_grid=GridSearchCV(KNeighborsClassifier(),inp_params,verbose=False,refit=True,cv=3)
        knn_grid.fit(X_train,y_train)
        knn_predict = knn_grid.predict(X_test)
        #self.printaccuracy(y_test,knn_predict,"KNN")
        #print("Best Hyperparameters " + str(knn_grid.best_params_) + " Best Score: " + str(knn_grid.best_score_))
        res= [knn_grid.score(X_train,y_train),
              knn_grid.score(X_test,y_test),
              precision_score(y_test,knn_predict,average ='weighted'),
              recall_score(y_test,knn_predict,average ='weighted'),f1_score(y_test,knn_predict,average ='weighted')]
        result = pd.DataFrame(np.array(res).reshape(-1,5))
        return result
    
    def logisticRegression(self,X_train,y_train,X_test,y_test,inp_params):
        print("Logistic Regression classification Starting...")
#        score = []
#         for pen in penalty_reg:
#             for i in Co_reg:
#                 for it in max_iteration:
#                     clf = LogisticRegression(random_state=0, solver='liblinear', penalty=pen , C=i, max_iter=it).fit(X_train, y_train.values.ravel())
#                     score.append(clf.score(X_test, y_test.values.ravel()))
        lr = GridSearchCV(LogisticRegression(random_state=0),inp_params,verbose=False,refit=True,cv=3)
        lr.fit(X_train,y_train)
        lr_predict = lr.predict(X_test)
        #self.printaccuracy(y_test,lr_predict,"Logistic Regression")
        res= [lr.score(X_train,y_train),
        lr.score(X_test,y_test),
        precision_score(y_test,lr_predict,average ='weighted'),
        recall_score(y_test,lr_predict,average ='weighted'),f1_score(y_test,lr_predict,average ='weighted')]
        result = pd.DataFrame(np.array(res).reshape(-1,5))
        return result
        
    def svm_model(self,X_train,y_train,X_test,y_test,inp_params):
        print("SVM Classification Starting...")
        svm = SVC(kernel='rbf',random_state=0)	
        svm_grid = GridSearchCV(svm, inp_params, verbose=False, cv=3,return_train_score=True)
        svm_grid.fit(X_train,y_train)
        svm_predict = svm_grid.predict(X_test)
        #self.printaccuracy(y_test,svm_predict,"SVM")
        #print("Best Hyperparameters " + str(svm_grid.best_params_) + " Best Score: " + str(svm_grid.best_score_))
        res= [svm_grid.score(X_train,y_train),
           svm_grid.score(X_test,y_test),
           precision_score(y_test,svm_predict,average ='weighted'),
           recall_score(y_test,svm_predict,average ='weighted'),f1_score(y_test,svm_predict,average ='weighted')]
        result = pd.DataFrame(np.array(res).reshape(-1,5))
        return result

    def decisionTreeClassifier(self,X_train,y_train,X_test,y_test,inp_params):
        print("Decisiontree Classifier Starting...")
        decisionTree_grid = GridSearchCV(DecisionTreeClassifier(random_state=0), inp_params, verbose=False, cv=3,return_train_score=True)
        decisionTree_grid.fit(X_train,y_train)
        decisionTree_predict = decisionTree_grid.predict(X_test)
        #self.printaccuracy(y_test,decisionTree_predict,"DecisionTree")
        #print("Best Hyperparameters " + str(decisionTree_predict.best_params_) + " Best Score: " + str(decisionTree_predict.best_score_))
        res= [decisionTree_grid.score(X_train,y_train),
           decisionTree_grid.score(X_test,y_test),
           precision_score(y_test,decisionTree_predict,average ='weighted'),
           recall_score(y_test,decisionTree_predict,average ='weighted'),f1_score(y_test,decisionTree_predict,average ='weighted')]
        result = pd.DataFrame(np.array(res).reshape(-1,5))
        return result
    
    def randomForest(self,X_train,y_train,X_test,y_test,inp_params):
        print("randomForest Classifier Starting...")
        rf = RandomForestClassifier(random_state=0)
        rf_grid = GridSearchCV(rf, inp_params, verbose=False, cv=3)
        rf_grid.fit(X_train,y_train)
        rf_predict = rf_grid.predict(X_test)
        #self.printaccuracy(y_test,rf_predict,"RandomForest")
        #print("Best Hyperparameters " + str(rf_grid.best_params_) + " Best Score: " + str(rf_grid.best_score_))
        res=[ rf_grid.score(X_train,y_train),
           rf_grid.score(X_test,y_test),
           precision_score(y_test,rf_predict,average ='weighted'),
           recall_score(y_test,rf_predict,average ='weighted'),f1_score(y_test,rf_predict,average ='weighted')]
        result = pd.DataFrame(np.array(res).reshape(-1,5))
        return result

    def adaBoost(self,X_train,y_train,X_test,y_test,inp_params):
        print("AdaBoost Classifier Starting...")
        ab = AdaBoostClassifier(random_state=0)
        ab_grid = GridSearchCV(ab, inp_params, verbose=False, cv=3)
        ab_grid.fit(X_train,y_train)
        ab_predict = ab_grid.predict(X_test)
        #self.printaccuracy(y_test,ab_predict,"AdaBoost")
        #print("Best Hyperparameters " + str(ab_grid.best_params_) + " Best Score: " + str(ab_grid.best_score_))
        res=[ab_grid.score(X_train,y_train),
           ab_grid.score(X_test,y_test),
           precision_score(y_test,ab_predict,average ='weighted'),
           recall_score(y_test,ab_predict,average ='weighted'),f1_score(y_test,ab_predict,average ='weighted')]
        result = pd.DataFrame(np.array(res).reshape(-1,5))
        return result
    
    def gaussianNaiveBaise(self,X_train,y_train,X_test,y_test):
        print("GaussianNaiveBaive Classifier Starting... ")
        gnb = GaussianNB()
        gnb.fit(X_train,y_train)
        gnb_predict = gnb.predict(X_test)
        #self.printaccuracy(y_test,gnb_predict,"Naive Bayes")
        res = [gnb.score(X_train,y_train),
           gnb.score(X_test,y_test),
           precision_score(y_test,gnb_predict,average ='weighted'),
           recall_score(y_test,gnb_predict,average ='weighted'),f1_score(y_test,gnb_predict,average ='weighted')]
        result = pd.DataFrame(np.array(res).reshape(-1,5))
        return result

    def neuralNetworks(self,X_train,y_train,X_test,y_test,inp_params):
        print("NeuralNetworks Classifier Starting...")
        nn = MLPClassifier(solver='sgd',random_state=0)
        nn_grid = GridSearchCV(nn, inp_params, cv=3)
        nn_grid.fit(X_train,y_train)
        nn_predict = nn_grid.predict(X_test)
        #self.printaccuracy(y_test,nn_predict,"Neural Networks")
        #print("Best Hyperparameters " + str(nn_grid.best_params_) + " Best Score: " + str(nn_grid.best_score_))
        res = [nn_grid.score(X_train,y_train),
           nn_grid.score(X_test,y_test),
           precision_score(y_test,nn_predict,average ='weighted'),
           recall_score(y_test,nn_predict,average ='weighted'),f1_score(y_test,nn_predict,average ='weighted')]
        result = pd.DataFrame(np.array(res).reshape(-1,5))
        return result

    def train_models(self,X_train,y_train,X_test,y_test,lr_params,knn_params,svm_params,decisiontree_params,random_forest_params,adaboost_params,nn_params,data):
        results = self.knn_grid_search(X_train,y_train,X_test,y_test,knn_params)
        results = pd.concat([results,self.svm_model(X_train,y_train,X_test,y_test,svm_params)])
        results = pd.concat([results,self.decisionTreeClassifier(X_train,y_train,X_test,y_test,decisiontree_params)])
        results = pd.concat([results,self.randomForest(X_train,y_train,X_test,y_test,random_forest_params)])
        results = pd.concat([results,self.adaBoost(X_train,y_train,X_test,y_test,adaboost_params)])
        results = pd.concat([results,self.logisticRegression(X_train,y_train,X_test,y_test,lr_params)])
        results = pd.concat([results,self.gaussianNaiveBaise(X_train,y_train,X_test,y_test)])
        results = pd.concat([results,self.neuralNetworks(X_train,y_train,X_test,y_test,nn_params)])
        A=[data,data,data,data,data,data,data,data]
        B =['Logistic Regression','KNN','SVM','Decision Tree','Random Forest','Adaboost','GaussionNB','Nueral Network']
        C=['Train Accuracy','Test Accuracy','Precision','Recall','F1 score']
        results.index = [A,B]
        results.columns=C
        return results

    def creditCardDataset(self):
        #For credit card Defaulters 
        df = pd.read_csv("../Datasets/Credit_card/credit.csv")
        df.drop(df.columns[0], axis=1, inplace=True)
        df.dropna(axis=0, inplace=True)
        df = df.iloc[1:]
        df = df.astype(float)
        df = self.removeoutliers(df,inplace=True)
        X = df.iloc[:,:23]
        y = df.iloc[:,23:24]
        X = self.normalizedata(X)
        X_train,X_test,y_train,y_test = self.train_split(X,y)
        lr_params = {'C':np.logspace(-4, 4, 20),
                    'penalty' :['l1','l2'],
                    'max_iter':[10,100,1000]}
        knn_params = {'n_neighbors':np.array(range(2,10))}
        svm_params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        decisiontree_params = {'max_depth' : np.arange(1, 10, 10),'min_samples_split': np.arange(0.1, 1.0, 10)}
        random_forest_params = {'n_estimators' : np.arange(10,100,10),'max_depth' : np.arange(1,6,2)}
        adaBoost_params = {'n_estimators' : np.arange(10,100,10)}
        nn_params = {'hidden_layer_sizes': np.arange(30,150,20),'learning_rate': ['constant','invscaling','adaptive'],'max_iter': np.arange(20,200,50)}
        creditCardDataset_results = self.train_models(X_train,y_train.values.ravel(),X_test,y_test.values.ravel(),lr_params,knn_params,svm_params,decisiontree_params,random_forest_params, adaBoost_params,nn_params,'Credit_Card')
        creditCardDataset_results
        return creditCardDataset_results


    def australianCredit(self):
        df = (pd.read_csv("../Datasets/Australian_Credit/australian.dat", sep='\s+', header=None))
        x = df.iloc[:,:14]
        y= df.iloc[:,14:15]
        x = (x - x.min())/(x.max() - x.min())
        df = pd.concat([x,y],axis=1)
        z = np.abs(stats.zscore(df))
        df = df[(z < 3).all(axis=1)]
        X = df.iloc[:,:14]
        y= df.iloc[:,14:15]
        X_train,X_test,y_train,y_test = self.train_split(X,y)
        lr_params = {'C':np.logspace(-4, 4, 20),
                    'penalty' :['l1','l2'],
                    'max_iter':[10,100,1000]}
        knn_params = {'n_neighbors':np.array(range(2,10))}
        svm_params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        decisiontree_params = {'max_depth' : np.arange(1, 10, 10),'min_samples_split': np.arange(0.1, 1.0, 10)}
        random_forest_params = {'n_estimators' : np.arange(10,100,10),'max_depth' : np.arange(1,6,2)}
        adaBoost_params = {'n_estimators' : np.arange(10,100,10)}
        nn_params = {'hidden_layer_sizes': np.arange(30,150,20),'learning_rate': ['constant','invscaling','adaptive'],'max_iter': np.arange(20,200,50)}
        australianCredit_results = self.train_models(X_train,
                                                     y_train.values.ravel(),
                                                     X_test,y_test.values.ravel(), 
                                                     lr_params,
                                                     knn_params,
                                                     svm_params,
                                                     decisiontree_params,
                                                     random_forest_params, 
                                                     adaBoost_params,
                                                     nn_params,
                                                    'Australia_Credit')
        return australianCredit_results
        
    def germanCredit(self):
        df = (pd.read_csv("../Datasets/German_credit_card/german.data-numeric", sep='\s+', header=None))
        df.dropna(inplace=True)
        X = df.iloc[:,:24]
        y = df.iloc[:,24:25]
        X = (X - X.min())/(X.max() - X.min())     
        df = pd.concat([X,y],axis=1)
        z = np.abs(stats.zscore(df))
        df = df[(z < 3).all(axis=1)]
        X = df.iloc[:,:24]
        y= df.iloc[:,24:25]
        X_train,X_test,y_train,y_test = self.train_split(X,y)
        lr_params = {'C':np.logspace(-4, 4, 20),
                    'penalty' :['l1','l2'],
                    'max_iter':[10,100,1000]}
        knn_params = {'n_neighbors':np.array(range(2,10))}
        svm_params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        decisiontree_params = {'max_depth' : np.arange(1, 10, 10),'min_samples_split': np.arange(0.1, 1.0, 10)}
        random_forest_params = {'n_estimators' : np.arange(10,100,10),'max_depth' : np.arange(1,6,2)}
        adaBoost_params = {'n_estimators' : np.arange(10,100,10)}
        nn_params = {'hidden_layer_sizes': np.arange(30,150,20),'learning_rate': ['constant','invscaling','adaptive'],'max_iter': np.arange(20,200,50)}
        germanCredit_results = self.train_models(X_train,y_train.values.ravel(),X_test,y_test.values.ravel(), 
                                                 lr_params,knn_params,svm_params,
                                                 decisiontree_params,random_forest_params, adaBoost_params,nn_params,'German_Credit')
        germanCredit_results
        return germanCredit_results

    def thoratic(self):
        data = pd.read_csv("../Datasets/9.ThoraticSurgeryData/ThoraricSurgery.arff",delimiter = ',',names=["DGN", "PRE4", "PRE5", "PRE6","PRE7","PRE8","PRE9","PRE10","PRE11","PRE14","PRE17","PRE19","PRE25","PRE30","PRE32","AGE","Risk1Y"])
        data.head()
        #Preprocessing
        X = pd.DataFrame(data,columns=["DGN", "PRE4", "PRE5", "PRE6","PRE7","PRE8","PRE9","PRE10","PRE11","PRE14","PRE17","PRE19","PRE25","PRE30","PRE32","AGE"])
        cat = ["DGN","PRE6","PRE7","PRE8","PRE9","PRE10","PRE11","PRE14","PRE17","PRE19","PRE25","PRE30","PRE32"]
        for i in cat:
            X[i] = pd.Categorical(X[i]).codes
        y = data.iloc[:,16:17]
        y['Risk1Y'] = pd.Categorical(y['Risk1Y']).codes
        X = self.normalizedata(X)
        X_train,X_test,y_train,y_test = self.train_split(X,y)
        lr_params = {'C':np.logspace(-4, 4, 20),
                    'penalty' :['l1','l2'],
                    'max_iter':[1000]}
        knn_params={'n_neighbors':np.array(range(2,50))}
        svm_params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        decisiontree_params = { 'max_depth' : np.arange(5,10,1)}
        random_forest_params = { 'n_estimators' : np.arange(10,100,10), 'max_depth' : np.arange(5,50,5)}
        adaBoost_params = { 'n_estimators' : np.arange(10,100,10)}
        nn_params = {
                    'hidden_layer_sizes': np.arange(50,150,20),
                    'learning_rate': ['constant','adaptive'],
                    'max_iter': np.arange(200,300,50)
                    }
        Thoratic_results = self.train_models(X_train,y_train.values.ravel(),X_test,y_test.values.ravel(),lr_params,knn_params,svm_params,decisiontree_params,random_forest_params, adaBoost_params,nn_params,'Thoracic Surgery Data')
        return Thoratic_results 
        
    def seismicbumps(self):
        data = pd.read_csv("../Datasets/SeismicBumps/seismic-bumps.arff",delimiter = ',',names=["seismic","seismoacoustic","shift","genergy","gpuls","gdenergy","gdpuls","ghazard","nbumps","nbumps2","nbumps3","nbumps4","nbumps5","nbumps6","nbumps7","nbumps89","energy","maxenergy","class"])
        data.head()
        #Preprocessing
        X = pd.DataFrame(data,columns=["seismic","seismoacoustic","shift","genergy","gpuls","gdenergy","gdpuls","ghazard","nbumps","nbumps2","nbumps3","nbumps4","nbumps5","nbumps6","nbumps7","nbumps89","energy","maxenergy"])
        cat = ["seismic","seismoacoustic","shift","ghazard"]
        for i in cat:
            X[i] = pd.Categorical(X[i]).codes
        y = data.iloc[:,18:29]
        y['class'] = pd.Categorical(y['class']).codes
        X = self.normalizedata(X)
        X_train,X_test,y_train,y_test = self.train_split(X,y)
        lr_params = {'C':np.logspace(-4, 4, 20),
                    'penalty' :['l1','l2'],
                    'max_iter':[1000]}
        knn_params={'n_neighbors':np.array(range(2,50))}
        svm_params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        decisiontree_params = { 'max_depth' : np.arange(5,10,1)}
        random_forest_params = { 'n_estimators' : np.arange(10,100,10), 'max_depth' : np.arange(5,50,5)}
        adaBoost_params = { 'n_estimators' : np.arange(10,100,10)}
        nn_params = {
                    'hidden_layer_sizes': np.arange(50,150,20),
                    'learning_rate': ['constant','adaptive'],
                    'max_iter': np.arange(200,300,50)
                    }

        Seismic_results = self.train_models(X_train,y_train.values.ravel(),X_test,y_test.values.ravel(),lr_params,knn_params,svm_params,decisiontree_params,random_forest_params, adaBoost_params,nn_params,
                                                    'Seismic-Bumps')
        return Seismic_results
        
    def steel_plates_faults(self):
        #Multiclass
        long_list= ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas', 'Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
        data = pd.read_csv("../Datasets/SteelPlatesFaults/Faults.NNA",delimiter = '\s+',names=long_list)
        X = pd.DataFrame(data,columns=['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas'])
        y = data.iloc[:,27:34]
        #Converting 7 columns into one y 'class' column
        def fun1(x):
            for i in range(len(x)):
                if x[i] == 1:
                    return i
        y1= []        
        for j in range(len(y)):        
            y1.append((fun1(y.iloc[j]))) 
        y2 = pd.DataFrame(y1)
        y2.columns=['Class']
        y=y2
        X = self.normalizedata(X)
        X_train,X_test,y_train,y_test = self.train_split(X,y)
        lr_params = {'C':np.logspace(-4, 4, 20),
                    'penalty' :['l1','l2'],
                    'max_iter':[1000]}
        knn_params={'n_neighbors':np.array(range(2,50))}
        svm_params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        decisiontree_params = { 'max_depth' : np.arange(5,10,1)}
        random_forest_params = { 'n_estimators' : np.arange(10,100,10), 'max_depth' : np.arange(5,50,5)}
        adaBoost_params = { 'n_estimators' : np.arange(10,100,10)}
        nn_params = {
                    'hidden_layer_sizes': np.arange(50,150,20),
                    'learning_rate': ['constant','adaptive'],
                    'max_iter': np.arange(200,300,50)
                    }

        Steel_Plates_Faults_results = self.train_models(X_train,y_train.values.ravel(),X_test,y_test.values.ravel(),lr_params,knn_params,svm_params,decisiontree_params,random_forest_params, adaBoost_params,nn_params,
                                                    'Steel_Plates_Faults')
        return Steel_Plates_Faults_results
        
    def diabetic_retinopaty(self):
        col = ['Quality','Pre-screening','MA1','MA2','MA3','MA4','MA5','MA6','MA7',
              'exudates1','exudates2','exudates3','exudates4','exudates5','exudates6','exudates7',
              'macula_opticdisc','opticdisc_diamter','AM/FM','Class_label']
        data = pd.read_csv('messidor_features.arff',error_bad_lines=False)
        data.columns = col
        data = data[data['Quality'] != 0]
        data.drop(['Quality','AM/FM','Pre-screening','MA1','MA2','MA3','MA5','MA6','exudates3','exudates4'
               ,'exudates6','exudates7'],axis=1,inplace=True)
        data = self.removeoutliers(data,inplace=True)
        X = data.drop('Class_label',axis=1)
        X_copy = X.copy()
        y= data['Class_label']
        X = self.normalizedata(X)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
        lr_params = {'C':np.logspace(-4, 4, 20),
                    'penalty' :['l1','l2'],
                    'max_iter':[1000]}
        knn_params={'n_neighbors':np.array(range(2,50))}
        svm_params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        decisiontree_params = { 'max_depth' : np.arange(5,10,1)}
        random_forest_params = { 'n_estimators' : np.arange(10,100,10), 'max_depth' : np.arange(5,10,1)}
        adaBoost_params = { 'n_estimators' : np.arange(10,100,10)}
        nn_params = {
                    'hidden_layer_sizes': np.arange(50,150,20),
                    'learning_rate': ['constant','adaptive'],
                    'max_iter': np.arange(200,300,50)
                    }
        diabetic_retinopaty_results = self.train_models(X_train,
                                                     y_train,
                                                     X_test,y_test, 
                                                     lr_params,
                                                     knn_params,
                                                     svm_params,
                                                     decisiontree_params,
                                                     random_forest_params, 
                                                     adaBoost_params,
                                                     nn_params,
                                                    'Diabetic Retinopathy')
        diabetic_retinopaty_results
        return diabetic_retinopaty_results
    
    def Breast_Cancer_Wisconsin(self):
        col = ['ID Number','Diagnosis','radius','texture','perimeter','area','smoothness','compactness','concavity','concave points','symmetry','fractal dimension'
        ,'radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave points2','symmetry2','fractal dimension2'
        ,'radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave points3','symmetry3','fractal dimension3']

        data = pd.read_csv('wdbc.data',error_bad_lines=False)
        data.columns = col
        data.drop('ID Number',axis=1,inplace=True)
        data['Diagnosis'] = pd.Categorical(data['Diagnosis']).codes
        data = self.removeoutliers(data,inplace=True)
        X = data.drop('Diagnosis',axis=1)
        X_copy = X.copy()
        y= data['Diagnosis']
        X = self.normalizedata(X)
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
        lr_params = {'C':np.logspace(-4, 4, 20),
                    'penalty' :['l1','l2'],
                    'max_iter':[1000]}
        knn_params={'n_neighbors':np.array(range(2,50))}
        svm_params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        decisiontree_params = { 'max_depth' : np.arange(5,10,1)}
        random_forest_params = { 'n_estimators' : np.arange(10,100,10), 'max_depth' : np.arange(5,50,5)}
        adaBoost_params = { 'n_estimators' : np.arange(10,100,10)}
        nn_params = {
                    'hidden_layer_sizes': np.arange(50,150,20),
                    'learning_rate': ['constant','adaptive'],
                    'max_iter': np.arange(200,300,50)
                    }
        Breast_Cancer_Wisconsin_results = self.train_models(X_train,
                                                     y_train,
                                                     X_test,y_test, 
                                                     lr_params,
                                                     knn_params,
                                                     svm_params,
                                                     decisiontree_params,
                                                     random_forest_params, 
                                                     adaBoost_params,
                                                     nn_params,
                                                    'Breast_Cancer_Wisconsin')
        return Breast_Cancer_Wisconsin_results
    
    
    def Adults_Salary(self):
        col = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex',
           'capital-gain','capital-loss','hours-per-week','native-country','salary']
        data = pd.read_csv("../Datasets/Adult_Salary_Data/adult.data")
        data.columns=col
        categorical_columns = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country','salary']
        for i in categorical_columns:
            data[i] = pd.Categorical(data[i]).codes
        data.columns = data.columns.str.lstrip()
        a = StandardScaler()
        X = data.drop(['salary'],axis=1)
        X = a.fit_transform(X)
        y = data['salary']
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)
        lr_params = {'C':np.logspace(-4, 4, 20),
                        'penalty' :['l1','l2'],
                        'max_iter':[1000]}
        knn_params={'n_neighbors':np.array(range(2,5))}
        svm_params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        decisiontree_params = { 'max_depth' : np.arange(5,10,1)}
        random_forest_params = { 'n_estimators' : np.arange(10,100,10), 'max_depth' : np.arange(5,50,5)}
        adaBoost_params = { 'n_estimators' : np.arange(10,100,10)}
        nn_params = {
                    'hidden_layer_sizes': np.arange(50,150,20),
                    'learning_rate': ['constant','adaptive'],
                    'max_iter': np.arange(200,300,50)
                    }
        adult_salary_results = self.train_models(X_train,
                                                     y_train,
                                                     X_test,y_test, 
                                                     lr_params,
                                                     knn_params,
                                                     svm_params,
                                                     decisiontree_params,
                                                     random_forest_params, 
                                                     adaBoost_params,
                                                     nn_params,
                                                    'Adult_Salary')
        return adult_salary_results
    
    
    def Yeast_Category(self):
        col = ['Sequence Name','mcg','gvh','alm','mit','erl','pox','vac','nuc','localization-site']
        data = pd.read_csv("../Datasets/Yeast_data/yeast.data",delim_whitespace=True)
        data.columns=col
        categorical_columns = ['Sequence Name','localization-site']
        for i in categorical_columns:
            data[i] = pd.Categorical(data[i]).codes
        data.columns = data.columns.str.lstrip()
        a = StandardScaler()
        X = data.drop(['localization-site'],axis=1)
        X = a.fit_transform(X)
        y = data['localization-site']
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3)
        lr_params = {'C':np.logspace(-4, 4, 20),
                            'penalty' :['l1','l2'],
                            'max_iter':[1000]}
        knn_params={'n_neighbors':np.array(range(2,5))}
        svm_params = { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        decisiontree_params = { 'max_depth' : np.arange(5,10,1)}
        random_forest_params = { 'n_estimators' : np.arange(10,100,10), 'max_depth' : np.arange(5,50,5)}
        adaBoost_params = { 'n_estimators' : np.arange(10,100,10)}
        nn_params = {
                    'hidden_layer_sizes': np.arange(50,150,20),
                    'learning_rate': ['constant','adaptive'],
                    'max_iter': np.arange(200,300,50)
                    }
        yeast_category_results = self.train_models(X_train,
                                                     y_train,
                                                     X_test,y_test, 
                                                     lr_params,
                                                     knn_params,
                                                     svm_params,
                                                     decisiontree_params,
                                                     random_forest_params, 
                                                     adaBoost_params,
                                                     nn_params,
                                                    'Yeast_Category')
        return yeast_category_results
    
    
    
    def getbest(self,data,name):
        temp = data.xs(name) 
        temp = temp[temp['F1 score'] == temp['F1 score'].max()]
        temp.index=[name+' - '+temp.index[0]]
        return temp
    


# In[17]:


entrypoint = entryPoint()

#GIRISH
temp = entrypoint.diabetic_retinopaty()
best_report = entrypoint.getbest(temp,'Diabetic Retinopathy')
Final_report = temp
temp = entrypoint.Breast_Cancer_Wisconsin()
best_report = pd.concat([best_report,entrypoint.getbest(temp,'Breast_Cancer_Wisconsin')])
Final_report = pd.concat([Final_report,temp])



# In[18]:


# ##Gursimran SIngh
temp = entrypoint.thoratic()
best_report = pd.concat([best_report,entrypoint.getbest(temp,'Thoracic Surgery Data')])
Final_report = pd.concat([Final_report,temp])

temp = entrypoint.seismicbumps()
best_report = pd.concat([best_report,entrypoint.getbest(temp,'Seismic-Bumps')])
Final_report = pd.concat([Final_report,temp])

temp = entrypoint.steel_plates_faults()
best_report = pd.concat([best_report,entrypoint.getbest(temp,'Steel_Plates_Faults')])
Final_report = pd.concat([Final_report,temp])


# In[19]:


# ##Aravind
temp = entrypoint.Adults_Salary()
best_report = pd.concat([best_report,entrypoint.getbest(temp,'Adult_Salary')])
Final_report = temp

temp = entrypoint.Yeast_Category()
best_report = pd.concat([best_report,entrypoint.getbest(temp,'Yeast_Category')])
Final_report = pd.concat([Final_report,temp])


# In[20]:


# ##Darshan
temp = entrypoint.creditCardDataset()
best_report = pd.concat([best_report,entrypoint.getbest(temp,'Credit_Card')])
Final_report = pd.concat([Final_report,temp])

temp = entrypoint.australianCredit()
best_report = pd.concat([best_report,entrypoint.getbest(temp,'Australia_Credit')])
Final_report = pd.concat([Final_report,temp])

temp = entrypoint.germanCredit()
best_report = pd.concat([best_report,entrypoint.getbest(temp,'German_Credit')])
Final_report = pd.concat([Final_report,temp])


print('Best Report')
best_report
print("  ")
print('Final Report')
Final_report

