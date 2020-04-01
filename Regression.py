#!/usr/bin/env python
import pandas as pd
import numpy as np
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

class RegressionModels():
    def train_split(self,X,y,test_size=0.2,random_state=0):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
        #scalar = StandardScaler().fit(X_train)
        #X_train = scalar.transform(X_train)
        #X_test = scalar.transform(X_test)
        #print("Normalization done")
        print("Test train split done")
        return X_train,X_test,y_train,y_test  

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
    
    def scores(self,y_test,y_pred, model):
        print("Variance Score : " , explained_variance_score(y_test, y_pred))
        print("R2 Score : " ,r2_score(y_test,y_pred))
        print("Root Mean Square : ",math.sqrt(mean_squared_error(y_test, y_pred)))
        print("Best Parameters : ", model.best_params_ )

    def svmRegression(self, X_train, X_test, y_train, Y_test, params):
        print("-----------SVM Regression Starts------------\n\n ")
        clf = SVR()
        svm_grid = GridSearchCV(clf, params, verbose=False, cv=3,return_train_score=True)
        svm_grid.fit(X_train,y_train)
        svm_predict = svm_grid.predict(X_test)
        res = [svm_grid.score(X_train,y_train),svm_grid.score(X_test,Y_test),
              r2_score(Y_test,svm_predict),math.sqrt(mean_squared_error(Y_test, svm_predict))]
        print("Best Parameters : ", svm_grid.best_params_ )
        result = pd.DataFrame(np.array(res).reshape(-1,4))
        return result
        self.scores(Y_test, svm_predict, svm_grid)
        print("\n-----------SVM Regression ends------------\n\n ")

    def decisionTreeRegression(self, X_train, X_test, y_train, Y_test, params):
        print("-----------Decision Tree Regression Starts------------\n\n")
        DTregressor = DecisionTreeRegressor(random_state=0)
        DT_grid = GridSearchCV(DTregressor, params, verbose=False, cv=3,return_train_score=True)
        DT_grid.fit(X_train,y_train)
        DT_predict = DT_grid.predict(X_test)
        res = [DT_grid.score(X_train,y_train),DT_grid.score(X_test,Y_test),
              r2_score(Y_test,DT_predict),math.sqrt(mean_squared_error(Y_test, DT_predict))]
        result = pd.DataFrame(np.array(res).reshape(-1,4))
        return result
        self.scores(Y_test, DT_predict, DT_grid)
        print("\n-----------Decision Tree Regression Ends------------\n\n")

    def randomForestRegression(self, X_train, X_test, y_train, Y_test, params):
        print("-----------Random Forest Regression Starts------------\n\n")
        RFRegressor = RandomForestRegressor(random_state=0)
        RF_grid = GridSearchCV(RFRegressor, params, verbose=False, cv=3, return_train_score=True)
        RF_grid.fit(X_train,y_train)
        RF_predict = RF_grid.predict(X_test)
        res = [RF_grid.score(X_train,y_train),RF_grid.score(X_test,Y_test),
              r2_score(Y_test,RF_predict),math.sqrt(mean_squared_error(Y_test, RF_predict))]
        result = pd.DataFrame(np.array(res).reshape(-1,4))
        return result
        self.scores(Y_test, RF_predict, RF_grid)
        print("\n-----------Random Forest Regression Ends------------\n\n")
  
    def adaBoostRegression(self, X_train, X_test, y_train, Y_test, params):
        print("-----------AdaBoost Regression Starts------------\n\n")
        AdaRegressor = AdaBoostRegressor(random_state=0)
        adaBoost_grid = GridSearchCV(AdaRegressor, params, verbose=False, cv=3,return_train_score=True)
        adaBoost_grid.fit(X_train,y_train)
        adaBoost_predict = adaBoost_grid.predict(X_test)
        res = [adaBoost_grid.score(X_train,y_train),adaBoost_grid.score(X_test,Y_test),
              r2_score(Y_test,adaBoost_predict),math.sqrt(mean_squared_error(Y_test, adaBoost_predict))]
        result = pd.DataFrame(np.array(res).reshape(-1,4))
        return result
        self.scores(Y_test, adaBoost_predict, adaBoost_grid)
        print("\n-----------AdaBoost Regression Ends------------\n\n")

    def gaussianProcessRegression(self, X_train, X_test, y_train, Y_test, params):
        print("-----------GaussianProcess Regression Starts------------\n\n")
        GPRRegressor = GaussianProcessRegressor(random_state=0)
        GPR_grid = GridSearchCV(GPRRegressor, params, verbose=False, cv=3,return_train_score=True)
        GPR_grid.fit(X_train,y_train)
        GPR_predict = GPR_grid.predict(X_test)
        res = [GPR_grid.score(X_train,y_train),GPR_grid.score(X_test,Y_test),
              r2_score(Y_test,GPR_predict),math.sqrt(mean_squared_error(Y_test, GPR_predict))]
        result = pd.DataFrame(np.array(res).reshape(-1,4))
        return result
        self.scores(Y_test, GPR_predict, GPR_grid)
        print("\n-----------GaussianProcess Regression Ends------------\n\n")

    def LinearRegression(self, X_train, X_test, y_train, Y_test, params):
        print("-----------Linear Regression Starts------------\n\n")
        LinearRegressor = LinearRegression()
        linearRegression_grid = GridSearchCV(LinearRegressor, params, verbose=False, cv=3,return_train_score=True)
        linearRegression_grid.fit(X_train,y_train)
        linearRegression_predict = linearRegression_grid.predict(X_test)
        res = [linearRegression_grid.score(X_train,y_train),linearRegression_grid.score(X_test,Y_test),
              r2_score(Y_test,linearRegression_predict),math.sqrt(mean_squared_error(Y_test, linearRegression_predict))]
        result = pd.DataFrame(np.array(res).reshape(-1,4))
        return result
        self.scores(Y_test, linearRegression_predict, linearRegression_grid)
        print("\n-----------Linear Regression Ends------------\n\n")

    def mlpRegression(self, X_train, X_test, y_train, Y_test, params):
        print("-----------Neural Network Regression Starts------------\n\n")
        MLPRegressor_obj = MLPRegressor(random_state=0)
        MLPRegressor_grid = GridSearchCV(MLPRegressor_obj, params, verbose=False, cv=3,return_train_score=True)
        MLPRegressor_grid.fit(X_train,y_train)
        MLPRegressor_predict = MLPRegressor_grid.predict(X_test)
        res = [MLPRegressor_grid.score(X_train,y_train),MLPRegressor_grid.score(X_test,Y_test),
              r2_score(Y_test,MLPRegressor_predict),math.sqrt(mean_squared_error(Y_test, MLPRegressor_predict))]
        result = pd.DataFrame(np.array(res).reshape(-1,4))
        return result
        self.scores(Y_test, MLPRegressor_predict, MLPRegressor_grid)
        print("\n-----------Neural Network Regression Ends------------\n\n")
     
    def train_all__models(self, X, y,data):
        svm_regression_params =  { 'C' : np.logspace(0, 3, 4), 'gamma' : np.logspace(-2, 1, 4)}
        dt_params = {'max_depth' : np.arange(1, 10, 10),'min_samples_split': np.arange(0.1, 1.0, 10)}
        rd_params = {'n_estimators' : np.arange(10,100,10),'max_depth' : np.arange(1,6,2)}
        ada_params = {'n_estimators' : np.arange(10,100,10)}
        gpr_params = {'n_restarts_optimizer' : np.arange(1,10,1)}
        linear_params = {'n_jobs' : np.arange(1,5,1)}
        mlp_params = {'hidden_layer_sizes': np.arange(30,150,20),'learning_rate': ['constant','invscaling','adaptive'],'max_iter':           np.arange(20,200,50)}
        X_train,X_test,y_train,y_test = self.train_split(X,y)
        
        results = self.svmRegression(X_train,X_test,y_train,y_test, svm_regression_params)   

        results = pd.concat([results,self.decisionTreeRegression(X_train,X_test,y_train,y_test, dt_params)])
        results = pd.concat([results,self.randomForestRegression(X_train,X_test,y_train,y_test, rd_params)])
        results = pd.concat([results,self.adaBoostRegression(X_train,X_test,y_train,y_test, ada_params)])
        results = pd.concat([results,self.gaussianProcessRegression(X_train,X_test,y_train,y_test, gpr_params)])
        results = pd.concat([results,self.LinearRegression(X_train,X_test,y_train,y_test, linear_params)])
        results = pd.concat([results,self.mlpRegression(X_train,X_test,y_train,y_test, mlp_params)])
        A=[data,data,data,data,data,data,data]
        B =['SVM Regression','Decision tree','Random forest','ADA boost','Gaussian process','Linear regression','MLP']
        C=['Train Accuracy','Test Accuracy','r2 score','Root mean square']
        results.index = [A,B]
        results.columns=C
        return results

    def wine_quality(self):
        df = pd.read_csv('winequality-red.csv',delimiter=';')
        df.dropna(axis=0,inplace=True)
        #print(df.corr()['quality'].drop('quality'))
        X = df[df.columns[0:11]]
        y = df[df.columns[11:12]]
        wine_quality_results = self.train_all__models(X, y.values.ravel(),'Wine_quality')
        print(wine_quality_results)
        return wine_quality_results
    
    def communities(self):
        columns_data = ['state','county','community','communityname','fold','population','householdsize','racepctblack','racePctWhite','racePctAsian','racePctHisp','agePct12t21',
                    'agePct12t29','agePct16t24','agePct65up','numbUrban','pctUrban','medIncome','pctWWage','pctWFarmSelf','pctWInvInc','pctWSocSec','pctWPubAsst','pctWRetire','medFamInc',
                    'perCapInc','whitePerCap','blackPerCap','indianPerCap','AsianPerCap','OtherPerCap','HispPerCap','NumUnderPov','PctPopUnderPov','PctLess9thGrade',
                    'PctNotHSGrad','PctBSorMore','PctUnemployed','PctEmploy','PctEmplManu','PctEmplProfServ','PctOccupManu','PctOccupMgmtProf','MalePctDivorce','MalePctNevMarr',
                    'FemalePctDiv','TotalPctDiv','PersPerFam','PctFam2Par','PctKids2Par','PctYoungKids2Par','PctTeen2Par','PctWorkMomYoungKids','PctWorkMom','NumIlleg','PctIlleg',
                    'NumImmig','PctImmigRecent','PctImmigRec5','PctImmigRec8','PctImmigRec10','PctRecentImmig','PctRecImmig5','PctRecImmig8','PctRecImmig10','PctSpeakEnglOnly','PctNotSpeakEnglWell'
                    ,'PctLargHouseFam','PctLargHouseOccup','PersPerOccupHous','PersPerOwnOccHous','PersPerRentOccHous','PctPersOwnOccup','PctPersDenseHous','PctHousLess3BR',
                    'MedNumBR','HousVacant','PctHousOccup','PctHousOwnOcc','PctVacantBoarded','PctVacMore6Mos','MedYrHousBuilt','PctHousNoPhone','PctWOFullPlumb','OwnOccLowQuart','OwnOccMedVal',
                    'OwnOccHiQuart','RentLowQ','RentMedian','RentHighQ','MedRent','MedRentPctHousInc','MedOwnCostPctInc','MedOwnCostPctIncNoMtg','NumInShelters','NumStreet','PctForeignBorn',
                    'PctBornSameState','PctSameHouse85','PctSameCity85','PctSameState85','LemasSwornFT','LemasSwFTPerPop','LemasSwFTFieldOps','LemasSwFTFieldPerPop','LemasTotalReq','LemasTotReqPerPop',
                    'PolicReqPerOffic','PolicPerPop','RacialMatchCommPol','PctPolicWhite','PctPolicBlack','PctPolicHisp','PctPolicAsian','PctPolicMinor','OfficAssgnDrugUnits','NumKindsDrugsSeiz',
                    'PolicAveOTWorked','LandArea','PopDens','PctUsePubTrans','PolicCars','PolicOperBudg','LemasPctPolicOnPatr','LemasGangUnitDeploy','LemasPctOfficDrugUn','PolicBudgPerPop',
                    'ViolentCrimesPerPop'
                    ]
        df = pd.read_csv('communities.data',delimiter=',',names=columns_data)

       
        df = df.replace('?',np.nan)
       
        df = df.drop(['fold','community','state','communityname','county'],axis=1)

       
        median_value = df.iloc[:,25].median(skipna = True)
        df.iloc[130,25] = median_value
        df = df.dropna(axis=1)
        X = df[df.columns[0:100]]
        y = df[df.columns[100:101]]
        
        results = self.train_all__models(X, y.values.ravel(),'communities')
        print(results)
        return results
    
    def speech_data(self):
        col = ['Subject_id','local_jitter','absolute_jitter','rap_jitter','ppq5_jitter','ddp_jitter','local_shimmer',
                       'db_shimmer','apq3_shimmer','apq5_shimmer','apq11_shimmer','dda_shimmer','AC','NTH','HTN','Median_pitch',
                       'Mean_pitch','Standard_deviation','Minimum_pitch','Maximum_pitch','Number_of_pulses','Number_of_periods',
                       'Mean_period','Standard_deviation_of_period','Fraction_of_locally_unvoiced_frames','Number_of_voice_breaks',
                       'Degree_of_voice_breaks','UPDRS','class_info']
        data = pd.read_csv("train_data.txt")
        data.columns=col

        data.columns = data.columns.str.lstrip()
        a = StandardScaler()
        X = data.drop(['class_info'],axis=1)
        X = a.fit_transform(X)
        y = data['class_info']
        results = self.train_all__models(X, y.values.ravel(),'speech data')
        print(results)
        return results
    
    def concrete_data(self):
        col = ['Cement','Blast Furnace Slag','Fly Ash','Water','Superplasticizer','Coarse Aggregate','Fine Aggregate',
                       'Age','Concrete compressive strength']
        
        data  = pd.read_excel("Concrete_Data.xls",skiprows=1)
        data.columns=col

        data.columns = data.columns.str.lstrip()
        a = StandardScaler()
        X = data.drop(['Concrete compressive strength'],axis=1)
        X = a.fit_transform(X)
        y = data['Concrete compressive strength']
        results = self.train_all__models(X, y.values.ravel(),'concrete_data')
        print(results)
        return results
        
    
    def student_data_train_G3(self):
        df1 = pd.read_csv("student-mat.csv",delimiter=";")
        df2 = pd.read_csv("student-por.csv",delimiter=";")

        data = pd.concat([df1,df2])

        categorical_columns = ['school','sex','famsize','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup',
                               'famsup','paid','activities','nursery','higher','internet','romantic']
        for i in categorical_columns:
            data[i] = pd.Categorical(data[i]).codes

        data.columns = data.columns.str.lstrip()
        a = StandardScaler()
       
        X = a.fit_transform(data)
        y = data['G3']
        results = self.train_all__models(X, y.values.ravel(),'student_data')
        print(results)
        return results
    
    def facebook(self):
        data = pd.read_csv('dataset_Facebook.csv',sep=';',error_bad_lines=False)
        data = data.dropna()
        data['Type'] = pd.Categorical(data['Type']).codes
        X = data[['Type','Category','Post Month','Post Weekday','Post Hour','Paid']]
        y = data['Lifetime People who have liked your Page and engaged with your post']
        results = self.train_all__models(X, y,'facebook')
        print(results)
        return results
    
    def qsar_aquatic_toxicity(self):
        col = ['TPSA(Tot)','SAacc','H-050','MLOGP','RDCHI','GATS1p','nN','C-040','quantitative response']
        data = pd.read_csv('qsar_aquatic_toxicity.csv',sep=';',error_bad_lines=False)
        data.columns = col
        data = data.dropna()
        X = data[['TPSA(Tot)','SAacc','H-050','MLOGP','RDCHI','GATS1p','nN','C-040']]
        y = data['quantitative response']
        results = self.train_all__models(X, y,'qsar_aquatic_toxicity')
        print(results)
        return results
  
    def bikesharing(self):
        data = pd.read_csv('hour.csv',error_bad_lines=False)
        data = data.dropna()
        X = data.drop(['instant','dteday','casual','registered','cnt'],axis=1,inplace=False)
        y = data['cnt']
        results = self.train_all__models(X, y,'bike_sharing')
        print(results)
        return results

    def merck_molecular_challenge(self):
        MERCK_FILE= '../Datasets/MerckActivity/TrainingSet/ACT2_competition_training.csv'
        with open(MERCK_FILE) as f:
            cols = f.readline().rstrip('\n').split(',') # Read the header line and get list of column names
            # Load the actual data, ignoring first column and using second column as targets.
            X = np.loadtxt(MERCK_FILE, delimiter=',', usecols=range(2, len(cols)), skiprows=1, dtype=np.uint8) 
            y = np.loadtxt(MERCK_FILE, delimiter=',', usecols=[1], skiprows=1)

        MERCK_FILE2= '../Datasets/MerckActivity/TrainingSet/ACT4_competition_training.csv'
        with open(MERCK_FILE2) as f:
            cols = f.readline().rstrip('\n').split(',') # Read the header line and get list of column names
            # Load the actual data, ignoring first column and using second column as targets.
            X_ACT4 = np.loadtxt(MERCK_FILE2, delimiter=',', usecols=range(2, len(cols)), skiprows=1, dtype=np.uint8) 
            y_ACT4 = np.loadtxt(MERCK_FILE2, delimiter=',', usecols=[1], skiprows=1)

        #saving into files ACT2
        from tempfile import TemporaryFile
        outfileACT2 = TemporaryFile()
        np.savez(outfileACT2, x_ACT2 = X, y_ACT2 = y)
        _ = outfileACT2.seek(0) 

        #saving into files ACT4
        outfileACT4 = TemporaryFile()
        np.savez(outfileACT4, x_ACT4 = X_ACT4, y_ACT4 = y_ACT4)
        _ = outfileACT4.seek(0) 

        # loading file ACT2
        npzfile_ACT2 = np.load(outfileACT2, allow_pickle=True)
        #print(npzfile_ACT2.files)

        # loading file ACT2
        npzfile_ACT4 = np.load(outfileACT4, allow_pickle=True)
        #print(npzfile_ACT4.files)

        # Dataframe
        X_ACT2 = pd.DataFrame(npzfile_ACT2['x_ACT2'])
        y_ACT2 = pd.DataFrame(npzfile_ACT2['y_ACT2'])

        X_ACT4 = pd.DataFrame(npzfile_ACT4['x_ACT4'])
        y_ACT4 = pd.DataFrame(npzfile_ACT4['y_ACT4'])
        
        results2 = self.train_all__models(X_ACT4, y_ACT4,'merk_molecular_challenge')
        
        return results2
        


    def SGEMM_GPU_kernel_performance(self):
        long_list = pd.read_csv("sgemm_product.csv",delimiter=',',nrows=0)
        long_list = ['MWG',	'NWG',	'KWG',	'MDIMC',	'NDIMC',	'MDIMA',	'NDIMB',	'KWI',	'VWM',	'VWN',	'STRM',	'STRN',	'SA',	'SB',	'Run1 (ms)',	'Run2 (ms)',	'Run3 (ms)',	'Run4 (ms)']
        data = pd.read_csv("sgemm_product.csv",delimiter = ',',skiprows=1,names=long_list)

        X = pd.DataFrame(data,columns=['MWG','NWG','KWG','MDIMC','NDIMC','MDIMA','NDIMB','KWI','VWM','VWN','STRM','STRN','SA','SB'])
        y = pd.DataFrame(data,columns=['Run1 (ms)'])
        results = self.train_all__models(X, y,'GPU_KERNEL_PERFORMANCE')
        print(results)
        return results
        
    def getbest(self,data,name):
        temp = data.xs(name) 
        temp = temp[temp['r2 score'] == temp['r2 score'].max()]
        temp.index=[name+' - '+temp.index[0]]
        return temp
        

regressionModels = RegressionModels()
temp = regressionModels.wine_quality()
best_report = regressionModels.getbest(temp,'Wine_quality')
Final_report = temp
temp = regressionModels.communities()
best_report = pd.concat([best_report,regressionModels.getbest(temp,'communities')])
Final_report = pd.concat([Final_report,temp])
temp = regressionModels.speech_data()
#best_report = pd.concat([best_report,regressionModels.getbest(temp,'speech data')])
Final_report = pd.concat([Final_report,temp])
temp = regressionModels.concrete_data()
best_report = pd.concat([best_report,regressionModels.getbest(temp,'concrete_data')])
Final_report = pd.concat([Final_report,temp])
temp = regressionModels.student_data_train_G3()
best_report = pd.concat([best_report,regressionModels.getbest(temp,'student_data')])
Final_report = pd.concat([Final_report,temp])
temp = regressionModels.facebook()
best_report = pd.concat([best_report,regressionModels.getbest(temp,'facebook')])
Final_report = pd.concat([Final_report,temp])
temp = regressionModels.qsar_aquatic_toxicity()
best_report = pd.concat([best_report,regressionModels.getbest(temp,'qsar_aquatic_toxicity')])
Final_report = pd.concat([Final_report,temp])
temp = regressionModels.bikesharing()
#best_report = pd.concat([best_report,regressionModels.getbest(temp,'bike_sharing')])
Final_report = pd.concat([Final_report,temp])
temp = regressionModels.merck_molecular_challenge()
best_report = pd.concat([best_report,regressionModels.getbest(temp,'merk_molecular_challenge')])
Final_report = pd.concat([Final_report,temp])
temp = regressionModels.SGEMM_GPU_kernel_performance()
best_report = pd.concat([best_report,regressionModels.getbest(temp,'GPU_KERNEL_PERFORMANCE')])
Final_report = pd.concat([Final_report,temp])

print("best_report")
print(best_report)
print("Final report")
print(Final_report)

