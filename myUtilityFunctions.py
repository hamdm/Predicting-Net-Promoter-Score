import pandas as pd
import numpy as np


def uniquevalues(df):
    cat_columns = ['MaritalStatus','Sex','BedCategory','Department','InsPayorcategory']
    for col in cat_columns:
        print('The unique values in {} are: {} and includes {}'.format(col,df[col].nunique(),
              np.unique(df[col].values,return_counts=True)),'\n')
        
def read_into_df(filename,sheetname,droplabels):
    data = pd.read_excel(filename,sheetname,index_col=0)
    data.drop(labels=droplabels,axis=1,inplace=True)
    return data
    
        
def missing_categories(df1,df2):
    cat_columns = ['MaritalStatus','Sex','BedCategory','Department','InsPayorcategory']
    for col in cat_columns:
        if df1[col].nunique != df2[col].nunique():
            missing_values=set(np.unique(df1[col].values))-set(np.unique(df2[col].values))
            mydict = {}
            for value in missing_values:
                count = df1[col].value_counts()[value]
                mydict[value] = count
        print("The categories not present in test data column {} are: {}, and their count in train data is {}".
              format(col,missing_values,mydict),'\n')
        
def extractFeatureTarget(df):
    features = df.iloc[:,:-1]
    target = df.iloc[:, df.columns.get_loc('NPS_Status')]
    return features,target

def concatenate_get_dummies(trainData,testData,axis):
    cat_columns = ['MaritalStatus','Sex','BedCategory','Department','InsPayorcategory']
    training_length = len(trainData)
    dataset = pd.concat(objs=[trainData,testData],axis=axis)
    dataset = pd.get_dummies(dataset,columns=cat_columns,prefix=cat_columns,drop_first=True)
    train_preprocessed = dataset[:training_length]
    test_preprocessed = dataset[training_length:]
    return train_preprocessed,test_preprocessed

def convToArray(df_X,df_y):
    features = df_X.columns
    X = df_X.values
    y = df_y.values
    return X,y,features
    #return X,labelEncoding(y)

def labelEncoding(le):
        labelencoder = LabelEncoder()
        le = labelencoder.fit_transform(le)
        return le
    


