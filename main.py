import os
#os.chdir('D:/capstone')

import pandas as pd
import numpy as np
from myUtilityFunctions import *
from raw_classifiers_nps import *
from applyingFeatureExtraction import *
from gridSearchCV_NPS import *
from gridSearchCV_NPS_Function import *
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC

'''
Categorical Columns not being used inside paranthesis:
(HospitalNo2),(STATEZONE),MaritalStatus, Sex, BedCategory, Department, 
InsPayorCategory, (State), (Country),(AdmissionDate),(DischargeDate), 
NPS_Status
'''    


'''
Reading Excel, dropping columns, and checking unique values in categorical 
columns of interest
'''
cat_columns = ['MaritalStatus','Sex','BedCategory','Department','InsPayorcategory']
labels_drop=['HospitalNo2','State','STATEZONE','Country','AdmissionDate','DischargeDate']
filename = 'ManipalHospitalExcel.xlsx'
training_sheet = 'Training Data or Binary Class'
test_sheet = 'Test Data for Binary Class'

#df = pd.read_excel('ManipalHospitalExcel.xlsx','Training Data or Binary Class',index_col=0)
#df_test = pd.read_excel('ManipalHospitalExcel.xlsx','Test Data for Binary Class',index_col=0)
#df.drop(labels=['HospitalNo2','State','STATEZONE','Country','AdmissionDate','DischargeDate'],axis=1,inplace=True)
#df_test.drop(labels=['HospitalNo2','State','STATEZONE','Country','AdmissionDate','DischargeDate'],axis=1,inplace=True)

df = read_into_df(filename,training_sheet,labels_drop)
df_test = read_into_df(filename,test_sheet,labels_drop)

missing_categories(df,df_test)

'''
The analysis above reveals that there are several categorical columns in the
test data that are missing values, hence I will concatenate the training data 
and the test data file against axis=0, use get_dummies, and then separate the 
two

'''

#Extracting Features,Target from both Train & Test Data
df_X,df_y = extractFeatureTarget(df)
df_X_test,df_y_test = extractFeatureTarget(df_test)


#Dummy Conversion
train_preprocessed,test_preprocessed = concatenate_get_dummies(df_X,df_X_test,axis=0)


#Converting to Array
X_train,y_train,features = convToArray(train_preprocessed,df_y)
X_test,y_test,features = convToArray(test_preprocessed,df_y_test)
X_train = np.delete(X_train,train_preprocessed.columns.get_loc('CE_NPS'),axis=1)
X_test = np.delete(X_test,train_preprocessed.columns.get_loc('CE_NPS'),axis=1)



training_accuracy,testing_accuracy,classifiers = classifier(X_train,X_test,y_train,y_test)

'''
plt.plot(training_accuracy,label="training_accuracy")
plt.plot(testing_accuracy,label='test_accuracy')
plt.xticks(range(len(classifiers)),classifiers,rotation=90)
'''

#Here we apply MinMaxScaler
X_train_scaled, X_test_scaled = apply_MinMax(X_train,X_test)
training_accuracy_MinMax,testing_accuracy_MinMax,classifiers = classifier(X_train_scaled,X_test_scaled,y_train,y_test)

'''
plt.plot(training_accuracy_MinMax,label="training_accuracy_scaled")
plt.plot(testing_accuracy_MinMax,label='test_accuracy_scaled')
plt.xticks(range(len(classifiers)),classifiers,rotation=90)
'''

#Checking Explained Variance with PCA

X_train_pca,X_test_pca,explained = explained_variance(X_train,X_test)

'''
plt.plot(explained[:10],label="Variance Explained")
plt.xticks(range(10),range(1,11),rotation=90)
'''

components_to_PCA(explained,0.60)


#Running classifiers with Principal Components
X_train_pca,X_test_pca = apply_PCA(X_train,X_test,13)
training_accuracy_pca,testing_accuracy_pca,classifiers = classifier(X_train_pca,X_test_pca,y_train,y_test)

'''
plt.plot(training_accuracy_pca,label="training_accuracy_scaled")
plt.plot(testing_accuracy_pca,label='test_accuracy_scaled')
plt.xticks(range(len(classifiers)),classifiers,rotation=90)
'''

training_accuracy_gridSearch,testing_accuracy_gridSearch,best_params_classifiers,classifiers = gridSearch(X_train,X_test,y_train,y_test)


'''
plt.plot(training_accuracy_gridSearch,label="training_accuracy_gridSearch")
plt.plot(testing_accuracy_gridSearch,label='test_accuracy_gridSearch')
plt.xticks(range(len(classifiers)),classifiers,rotation=90)
'''


training_accuracy_fs,testing_accuracy_fs,best_params_classifiers_fs,classifiers = gridSearch_FS(X_train,X_test,y_train,y_test)

'''
plt.plot(training_accuracy_fs,label="training_accuracy_FS")
plt.plot(testing_accuracy_fs,label='test_accuracy_FS')
plt.xticks(range(len(classifiers)),classifiers,rotation=90)
'''

